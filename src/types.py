import base64
import json
import os
import re
from functools import wraps
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4

import aiohttp
import tiktoken
from git import GitCommandError, Repo
from litellm import acompletion
from pydantic import BaseModel, Field, computed_field

from src.interpreter import Interpreter
from src.models import ModelNames, get_token_count
from utils.functions import FunctionSet, Payload, load_functions


def str_to_path(path: Union[str, List[str]]) -> Path:
    if isinstance(path, list):
        return Path(*path)
    return Path(path)


def add_token_count(cls):
    @property
    @wraps(cls)
    def tokens(self) -> int:
        return get_token_count(self.text, self.model_name)

    cls.tokens = tokens
    return cls


#############################################
## For all the Git Related Functions ########
#############################################
GitDiff = Union[
    Dict[str, Union[str, os.PathLike]], List[Dict[str, Union[str, os.PathLike]]]
]


def get_latest_changes(root_path: Union[str, os.PathLike]) -> Tuple[GitDiff, str]:
    """
    Get the latest changes within a git repository.

    Parameters
    ----------
    root_path : Union[str, os.PathLike]
        Path to the git repository.

    Returns
    -------
    Tuple[List[GitFileDiff], str]
        A tuple containing the original diffs and their summaries.
    """
    root_path = str_to_path(root_path)
    try:
        repo = Repo(root_path)
    except GitCommandError:
        raise ValueError("Invalid Git repository path")

    diffs = []
    summaries = []
    staged_files = [item.a_path for item in repo.index.diff("HEAD")]

    for file in staged_files:
        try:
            diff = repo.git.diff("HEAD", file)
            summary = summarize_diff(diff)
            diffs.append(GitDiff(file, diff))
            summaries.append(f"{file}: {summary}")
        except GitCommandError:
            pass

    summary_text = "\n".join(summaries)
    return diffs, summary_text


def summarize_diff(diff: str) -> str:
    """
    Generate a human-readable summary of the diff.

    Parameters
    ----------
    diff : str
        A string representing the diff output.

    Returns
    -------
    str
        A string containing a summary of the changes in the diff.
    """
    lines = diff.splitlines()
    added = sum(
        1 for line in lines if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in lines if line.startswith("-") and not line.startswith("---")
    )
    modified = len(lines) - added - removed

    description = []
    if added:
        description.append(f"{added} lines added")
    if removed:
        description.append(f"{removed} lines removed")
    if modified:
        description.append(f"{modified} lines modified")

    return ", ".join(description) if description else "No changes detected"


def get_git_commit(diff: GitDiff) -> "Conversation":
    """
    Get the git commit prompt.

    Parameters
    ----------
    diff : GitFileDiff
        The git file diff.

    Returns
    -------
    Conversation
        Git commit prompt.
    """
    # Start the system message with a list of rules, it will be further
    # appended depending on the code_only flag
    system_message = System(
        "\n\n".join(
            [
                "Your only task is to provide a very comprehensive git commit message. ",
                "Try and be as detailed as possible, format it within points if needed. ",
                "You will be provided with an object of the structure: ",
                f"Git Diff Struct:",
                json.dumps(GitDiff.__dict__()),
            ]
        ),
    )

    # Get the formatted messages
    user_message = User(
        "\n\n".join(
            [
                f"Here is the git diff structure between the <gitDiff></gitDiff> tags: ",
                f"<gitDiff>{diff}</gitDiff>",
                "Give me a very comprehensive git commit message, in markdown. ",
                "Explain the benefits of the changes, and the drawbacks of the changes. ",
                "If they're just formatting changes, then say so, be succinct when needed. ",
            ]
        ),
    )

    # Create the conversation object
    return Conversation(messages_=[system_message, user_message])


####################################################
########## LLM Function Calling Types ##############
####################################################
class CodeExecutionContent(BaseModel):
    type: Literal["code_execution"] = "code_execution"
    code: str  # The executed code
    stdout: Optional[str] = None  # Standard output from execution
    stderr: Optional[str] = None  # Standard error (if any)

    @computed_field
    @property
    def text(self) -> str:
        """
        Generates formatted Markdown text for code and output.

        Returns
        -------
        str
            The formatted Markdown text.
        """
        markdown_text = f"**Generated Code:**\n---\n```python\n{self.code} \n```\n\n"

        if self.stdout:
            markdown_text += f"**Output:**\n---\n```stdout\n{self.stdout} \n``` \n\n"

        if self.stderr:
            markdown_text += f"**Error:**\n---\n```stderr\n{self.stderr} \n``` \n\n"

        return markdown_text


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: Optional[str | Dict[str, str]]

    @classmethod
    def validate_image_path(cls, image_url: str) -> Dict[str, str]:
        """
        Validate the image path and encode the image in base64 format.

        Parameters
        ----------
        image_url : str
            The path to the image file.

        Returns
        -------
        Dict[str, str]
            A dictionary containing the base64-encoded image URL.
        """
        with open(image_url, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_url = {"url": f"data:image/jpeg;base64,{encoded_image}"}
        return image_url


class FunctionCallContent(BaseModel):
    type: Literal["function_call"] = "function_call"
    name: str  # The name of the function to call
    arguments: str  # JSON-formatted arguments for the function
    tool_call_id: Optional[str] = None  # Add the tool_call_id field

    def tokens(self, enc: tiktoken.Encoding) -> int:
        """
        Calculate the number of tokens in the function call content.

        Parameters
        ----------
        enc : tiktoken.Encoding
            The encoding to use for token counting.

        Returns
        -------
        int
            The number of tokens.
        """
        return len(enc.encode(self.name)) + len(enc.encode(self.arguments))


# TODO: Add support for other content types like images, etc.
@add_token_count
class User(BaseModel):
    role: str = "user"
    content: TextContent

    def __init__(self, msg: str, **data):
        super().__init__(content=TextContent(text=msg), **data)


@add_token_count
class Assistant(BaseModel):
    role: str = "assistant"
    content: Union[TextContent, CodeExecutionContent, FunctionCallContent]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def __init__(
        self,
        content: Union[TextContent, CodeExecutionContent, FunctionCallContent],
        **data,
    ):
        """Initialize Assistant with any of the allowed content types."""
        super().__init__(content=content, **data)


@add_token_count
class System(BaseModel):
    role: str = "system"
    content: TextContent

    def __init__(self, msg: str, **data):
        super().__init__(content=TextContent(text=msg), **data)


##################################################
########## Function Calling related ##############
##################################################
class FunctionExecutor:
    @staticmethod
    async def execute_code(
        code: str, interpreter: "Interpreter"
    ) -> AsyncGenerator["CodeExecutionContent", None]:
        code = code.lstrip("```python").rstrip("```")
        stdout, stderr = await interpreter.run(code)
        for line in stdout.splitlines():
            yield CodeExecutionContent(code=code, stdout=line, stderr="")
        if stderr:
            yield CodeExecutionContent(code=code, stdout="", stderr=stderr)

    @staticmethod
    async def analyze_data(
        file_path: str, instructions: str = None
    ) -> AsyncGenerator["TextContent", None]:
        # Simplified implementation for brevity
        analysis = f"Analysis of {file_path} with instructions: {instructions}"
        yield TextContent(text=analysis)

    @staticmethod
    async def git_operations(root_path: str) -> AsyncGenerator["TextContent", None]:
        diffs, diffsummary = get_latest_changes(root_path)
        yield TextContent(text=diffsummary)

    @staticmethod
    async def generate_code(
        instructions: str,
    ) -> AsyncGenerator["CodeExecutionContent", None]:
        # Simplified implementation for brevity
        generated_code = f"# Generated code based on: {instructions}"
        yield CodeExecutionContent(code=generated_code, stdout="", stderr="")


#############################################
############### Convo Class #################
#############################################
class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    model_name: str = Field(
        ..., default_factory=lambda: ModelNames.CLAUDE_3_5_SONNET.value
    )
    accumulated_arguments: Dict = Field(default={})
    active_function_calls: Dict = Field(default={})
    messages_: List[Union[User, Assistant, System]] = Field(default_factory=list)
    interpreter: Interpreter = Field(default_factory=Interpreter)
    files: List[str] = Field(default_factory=list)
    context_code: str = Field(default_factory=str)
    hist_path: str = Field(default="")
    code_path: str = Field(default="")
    functions_filepath: str | os.PathLike = Field(default="functions.json")
    functions: FunctionSet = Field(
        default_factory=lambda: load_functions("functions.json")
    )
    current_interaction: dict = Field(default_factory=dict)

    @property
    def encoding(self) -> tiktoken.Encoding:
        """
        Get the encoding for the model.

        Returns
        -------
        tiktoken.Encoding
            The encoding for the model.
        """
        return ModelNames.get_encoding(self.model_name)

    @property
    def total_tokens(self) -> int:
        """
        Calculate the total number of tokens in the conversation.

        Returns
        -------
        int
            The total number of tokens.
        """
        return get_token_count(self.messages_, self.model_name)

    @property
    def messages(self) -> List:
        """
        Get the list of messages in the conversation.

        Returns
        -------
        List[Union[User, Assistant, System]]
            The list of messages.
        """
        return self.messages_

    @property
    def to_dict(self) -> List[dict]:
        """
        Convert the conversation to a list of dictionaries.

        Returns
        -------
        List[dict]
            The list of dictionaries representing the conversation.
        """
        return [
            {
                "role": m.role,
                "content": (
                    m.content.text
                    if isinstance(m.content, TextContent)
                    else (
                        m.content.text
                        if isinstance(
                            m.content, (CodeExecutionContent, FunctionCallContent)
                        )
                        else str(m.content)
                    )
                ),
            }
            for m in self.messages_
        ]

    def __payload__(self, max_tokens: int, stream: bool = True) -> Payload:
        """
        Generate the payload for the LLM call.

        Parameters
        ----------
        max_tokens : int
            The maximum number of tokens.
        stream : bool
            Whether to stream the response.

        Returns
        -------
        Payload
            The payload for the LLM call.
        """
        return Payload.create(
            model=self.model_name,
            messages=self.to_dict,
            max_tokens=max_tokens,
            functions=self.functions,
            stream=stream,
        )

    def append(self, message: Union[User, Assistant, System]) -> None:
        """
        Append a message to the conversation.

        Parameters
        ----------
        message : Union[User, Assistant, System]
            The message to append.
        """
        self.messages_.append(message)

    def remove_file(self, filename: str) -> None:
        """
        Remove a file from the conversation and update the context code.

        Parameters
        ----------
        filename : str
            The name of the file to remove.
        """
        if filename in self.files:
            self.files.remove(filename)
            # Update the context_code by removing the content of the removed file
            self.update_context_code()

    def update_context_code(self) -> None:
        """
        Update the context code by concatenating the content of all files.
        """
        all_code = []
        for filename in self.files:
            with open(filename, "r", encoding="utf-8") as f:
                code = f.read()
                formatted_code = f"### filename: {filename} ###\n\n{code}\n\n###"
                all_code.append(formatted_code)

        self.context_code = "".join(all_code)

    def add_user_msg(self, msg: str) -> None:
        """
        Add a user message to the conversation.

        Parameters
        ----------
        msg : str
            The user message.
        """
        self.append(User(msg=msg))

    async def add_assistant_msg(
        self,
        content: Union[TextContent, CodeExecutionContent, FunctionCallContent] = None,
        **kwargs,  # Additional kwargs (e.g., name, tool_call_id)
    ) -> None:
        """
        Adds an assistant message to the conversation with flexible content types.

        Parameters
        ----------
        content : Union[TextContent, CodeExecutionContent, FunctionCallContent], optional
            The content of the assistant's message. Can be TextContent, CodeExecutionContent, or FunctionCallContent.
        **kwargs : dict
            Additional keyword arguments to pass to the Assistant constructor (e.g., name, tool_call_id).

        Raises
        ------
        ValueError
            If content is not provided.
        """
        if content is None:
            raise ValueError("Content must be provided.")

        assistant_message = Assistant(content=content, **kwargs)
        self.append(assistant_message)

    @staticmethod
    async def call_llm_no_context(messages: List[dict]):
        """
        Call the LLM without context.

        Parameters
        ----------
        messages : List[dict]
            The list of messages.

        Yields
        ------
        str
            The partial message from the LLM.
        """
        response = await acompletion(
            model="azure/PDFS-GPT-4o",
            messages=messages,
            max_tokens=4000,
            timeout=120,
            temperature=0.3,
            stream=True,
            api_base=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )
        partial_message = ""
        async for chunk in response:
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                partial_message += chunk.choices[0].delta.content
                yield partial_message

    async def call_llm(self, max_tokens: int, event_handler):
        """
        Call the LLM with context using the LiteLLM proxy.

        Parameters
        ----------
        max_tokens : int
            The maximum number of tokens.
        event_handler : ChainlitEventHandler
            The event handler to handle the response chunks.
        """
        payload = self.__payload__(max_tokens)
        url = payload.api_base + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {payload.api_key}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload.model_dump(exclude=["api_key", "api_base"]),
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        await event_handler.handle_sse_line(
                            line.decode("utf-8").strip()
                        )
                else:
                    error_text = await response.text()
                    print(f"Error calling LLM: HTTP {response.status}, {error_text}")

    def init_log(self) -> None:
        """
        Initialize the logger files.
        """
        if self.hist_path:
            with open(self.hist_path, "w") as file:
                json.dump([], file)

        if self.code_path:
            with open(self.code_path, "w") as file:
                json.dump([], file)

    def log_user_message(self, user_message: str) -> None:
        """
        Log a user message.

        Parameters
        ----------
        user_message : str
            The user message to log.
        """
        self.current_interaction["user_message"] = user_message

    def log_chat_message(self, chat_message: str) -> None:
        """
        Log a chat message.

        Parameters
        ----------
        chat_message : str
            The chat message to log.
        """
        self.current_interaction["chat_message"] = chat_message
        self.save_convo()

    def save_convo(self) -> None:
        """
        Save the current interaction to the history file.
        """
        if os.path.exists(self.hist_path):
            with open(self.hist_path, "r") as file:
                data = json.load(file)
        else:
            data = []

        # Append the new interaction
        data.append(self.current_interaction)

        with open(self.hist_path, "w") as file:
            json.dump(data, file, indent=4)

        self.current_interaction = {}

    def get_current_log(self) -> List[dict]:
        """
        Get the current log from the history file.

        Returns
        -------
        List[dict]
            The current log.
        """
        if os.path.exists(self.hist_path):
            with open(self.hist_path, "r") as file:
                data = json.load(file)
            return data
        else:
            return []

    def extract_and_append_code(self, chat_message) -> None:
        """
        Extract code blocks from the chat message and append them to the code file.

        Parameters
        ----------
        chat_message : str
            The chat message containing code blocks.
        """
        code_block_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        code_blocks = code_block_pattern.findall(chat_message)

        if not os.path.exists(self.code_path):
            with open(self.code_path, "w") as f:
                json.dump([], f)

        # Load the existing data from the JSON file
        with open(self.code_path, "r") as f:
            data = json.load(f)

        for block in code_blocks:
            data.append({"code": block.strip()})

        with open(self.code_path, "w") as f:
            json.dump(data, f, indent=4)

    async def execute_function(
        self, function_name: str, *args, **kwargs
    ) -> AsyncGenerator[Any, None]:
        executor = FunctionExecutor.get_executor(function_name)
        async for result in executor.execute(*args, **kwargs):
            yield result
