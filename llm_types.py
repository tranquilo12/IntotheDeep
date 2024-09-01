import base64
import json
import os
import re
from typing import Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4

import aiohttp
import tiktoken
from litellm import acompletion
from pydantic import BaseModel, Field, computed_field

from models import ModelNames


#############################################
########## Interpreter related ##############
#############################################
class Interpreter(BaseModel):
    endpoint: str = Field(default="http://localhost:8888/execute")

    class Config:
        arbitrary_types_allowed = True  # Allow the aiohttp ClientSession

    async def run(self, code: str) -> Tuple[str, str]:
        """
        Execute the provided code using an HTTP POST request to the specified endpoint.

        Parameters
        ----------
        code : str
            The code to execute.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the standard output and standard error from the code execution.
        """
        session_timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            async with session.post(self.endpoint, json={"code": code}) as response:
                result = await response.text()
                result = json.loads(result)
                return result["stdout"], result["stderr"]


def get_token_count(text: str, model_name: str) -> int:
    model_type = ModelNames.get_model_type(model_name)

    if model_type == "OPENAI" or model_type == "GGUF":
        encoding = tiktoken.encoding_for_model(
            "gpt-3.5-turbo" if "35" in model_name else "gpt-4"
        )
        return len(encoding.encode(text))
    elif model_type == "ANTHROPIC":
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    else:
        raise ValueError(f"Unknown model type: {model_name}")


class BaseModelsTokenCount(BaseModel):
    @property
    def tokens(self) -> int:
        """
        Calculate the number of tokens in the generated Markdown text.
        Returns
        -------
        int
            The number of tokens.
        """
        return get_token_count(self.text, self.model_name)


class CodeExecutionContent(BaseModelsTokenCount):
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


class TextContent(BaseModelsTokenCount):
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


class User(BaseModelsTokenCount):
    role: str = "user"
    content: TextContent

    def __init__(self, msg: str, **data):
        super().__init__(content=TextContent(text=msg), **data)


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


class System(BaseModel):
    role: str = "system"
    content: TextContent

    def __init__(self, msg: str, **data):
        super().__init__(content=TextContent(text=msg), **data)


#############################################
############### Convo Class #################
#############################################
class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    model_name: str = Field(
        ..., default_factory=lambda: ModelNames.AZURE_GPT_35_TURBO.value
    )
    accumulated_arguments: Dict = Field(default={})
    active_function_calls: Dict = Field(default={})
    messages_: List[Union[User, Assistant, System]] = Field(default_factory=list)
    interpreter: Interpreter = Field(default_factory=Interpreter)
    files: List[str] = Field(default_factory=list)
    context_code: str = Field(default_factory=str)
    hist_path: str = Field(default="")
    code_path: str = Field(default="")
    current_interaction: dict = Field(default_factory=dict)

    @property
    def model_name_enc(self) -> str:
        """
        Get the model name encoding.

        Returns
        -------
        str
            The model name encoding.
        """
        model_type = ModelNames.get_model_type(self.model_name)
        if model_type == "OPENAI":
            return "gpt-3.5-turbo" if "3.5" in self.model_name else "gpt-4"
        elif model_type == "ANTHROPIC":
            return self.model_name
        elif model_type == "LMSTUDIO":
            return (
                "gpt-3.5-turbo"  # Assuming local models use OpenAI-compatible tokenizer
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")

    @property
    def encoding(self) -> tiktoken.Encoding:
        """
        Get the encoding for the model.

        Returns
        -------
        tiktoken.Encoding
            The encoding for the model.
        """
        return tiktoken.encoding_for_model(self.model_name_enc)

    @property
    def total_tokens(self) -> int:
        """
        Calculate the total number of tokens in the conversation.

        Returns
        -------
        int
            The total number of tokens.
        """
        return sum(
            get_token_count(msg.content.text, self.model_name) for msg in self.messages_
        )

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

    def __payload__(self, max_tokens: int, stream: bool = True) -> dict:
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
        dict
            The payload for the LLM call.
        """

        base_payload = {
            "model": self.model_name,
            "messages": self.to_dict,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": stream,
            "function_call": "auto",
            "functions": [
                {
                    "name": "execute_code_locally",
                    "description": "Execute provided Python code locally and return the standard output and standard error.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute, enclosed in triple backticks (```python) and (```).",
                            }
                        },
                        "required": ["code"],
                    },
                },
                {
                    "name": "generate_code",
                    "description": "Generate Python code according to the past conversation and instructions provided by the user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instructions": {
                                "type": "string",
                                "description": "The complete set of instructions required to generate the expected code, including relevant past messages and user instructions",
                            }
                        },
                        "required": ["instructions"],
                    },
                },
                {
                    "name": "debug_code",
                    "description": "Debugs the Python code provided and return corrected code and output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute, enclosed in triple backticks (```python) and (```).",
                            }
                        },
                        "required": ["code"],
                    },
                },
                {
                    "name": "data_analyst",
                    "description": "Perform data analysis on the provided data file and return data insights and analysis results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Filepath to the data",
                            },
                            "instructions": {
                                "type": "string",
                                "description": "User instructions for the analysis",
                            },
                        },
                        "required": ["file_path", "instructions"],
                    },
                },
                {
                    "name": "data_generator",
                    "description": "Generates code for synthetic data generation based on a prior data analysis. It requires the data_analysis function to be called before invoking the data_generator function. If the analysis results are not available, ensure that data_analysis is executed to obtain the necessary data insights.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "instructions": {
                                "type": "string",
                                "description": "Complete analysis output after calling the data analyst function. Do not modify the analysis.",
                            }
                        },
                        "required": ["instructions"],
                    },
                },
                {
                    "name": "get_latest_changes_within_git",
                    "description": "Get the latest changes within a git repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "root_path": {
                                "type": "string",
                                "description": "Path to the git repository",
                            }
                        },
                        "required": ["root_path"],
                    },
                },
                {
                    "name": "get_git_commit_prompt",
                    "description": "Get the git commit prompt",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "root_path": {
                                "type": "string",
                                "description": "Path to the git repository",
                            }
                        },
                        "required": ["root_path"],
                    },
                },
            ],
        }

        return base_payload

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
        Call the LLM with context.

        Parameters
        ----------
        max_tokens : int
            The maximum number of tokens.
        event_handler : object
            The event handler to handle the response chunks.
        """
        payload = self.__payload__(max_tokens)
        model_type = ModelNames.get_model_type(self.model_name)
        api_base = ModelNames.get_api_base(self.model_name)

        try:
            response = await acompletion(
                **payload,
                api_base=(
                    os.getenv(api_base)
                    if api_base != "http://localhost:1234/v1"
                    else api_base
                ),
                api_key=os.getenv(f"{model_type.upper()}_API_KEY"),
            )
            async for chunk in response:
                await event_handler.handle_chunk(chunk)  # type: ignore
        except Exception as e:
            print(f"Error calling LLM: {str(e)}")

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
