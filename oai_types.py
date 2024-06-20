import base64
import json
import logging
from typing import Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4

import aiohttp
import tiktoken
from litellm import acompletion
from pydantic import BaseModel, Field, computed_field


#############################################
########## Interpreter related ##############
#############################################
class Interpreter(BaseModel):
    endpoint: str = Field(default="http://localhost:8888/execute")

    class Config:
        arbitrary_types_allowed = True  # Allow the aiohttp ClientSession

    async def run(self, code: str) -> Tuple[str, str]:
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json={"code": code}) as response:
                result = await response.text()
                result = json.loads(result)
                return result["stdout"], result["stderr"]


async def execute_code_locally(code: str, interpreter: Interpreter):
    """Executes Python code using the Interpreter and yields CodeExecutionContent chunks."""
    stdout, stderr = await interpreter.run(code)

    # Split the stdout into lines and yield each line as a separate chunk
    for line in stdout.splitlines():
        yield CodeExecutionContent(code=code, stdout=line, stderr="")

    # If there's any stderr, yield it as a final chunk
    if stderr:
        yield CodeExecutionContent(code=code, stdout="", stderr=stderr)


#############################################
########## LLM related types ################
#############################################
class CodeExecutionContent(BaseModel):
    type: Literal["code_execution"] = "code_execution"
    code: str  # The executed code
    stdout: str  # Standard output from execution
    stderr: Optional[str] = None  # Standard error (if any)

    @computed_field
    @property
    def text(self) -> str:
        """Generates formatted Markdown text for code and output."""
        markdown_text = f"**Generated Code**\n---\n```python\n{self.code}\n```\n"

        if self.stdout:
            markdown_text += f"**Output:**\n---\n```stdout\n{self.stdout}\n```\n\n"

        if self.stderr:
            markdown_text += f"**Error:**\n---\n```stderr\n{self.stderr}\n```\n\n"

        return markdown_text

    def tokens(self, enc: tiktoken.Encoding) -> int:
        """

        Parameters
        ----------
        enc :

        Returns
        -------

        """
        # Use the generated markdown text for token counting
        return len(enc.encode(self.text))


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

    def tokens(self, enc: tiktoken.Encoding) -> int:
        """

        Parameters
        ----------
        enc :

        Returns
        -------

        """
        return len(enc.encode(self.text))


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: Optional[str | Dict[str, str]]

    @classmethod
    def validate_image_path(cls, image_url: str) -> object:
        """

        Parameters
        ----------
        image_url :

        Returns
        -------

        """
        with open(image_url, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_url = {"url": f"data:image/jpeg;base64,{encoded_image}"}
        return image_url

    def tokens(self, enc: tiktoken.Encoding) -> int:
        """

        Parameters
        ----------
        enc :

        Returns
        -------

        """
        return len(enc.encode(self.image_url))


class User(BaseModel):
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
    model_name: str
    accumulated_arguments: Dict = Field(default={})
    active_tool_calls: Dict = Field(default={})
    messages_: List[Union[User, Assistant, System]] = Field(default_factory=list)
    interpreter: Interpreter = Field(default_factory=Interpreter)

    @property
    def token_encoding(self) -> tiktoken.Encoding:
        if self.model_name is not None:
            if "gpt" in self.model_name.lower():
                return tiktoken.encoding_for_model(self.model_name)

    @property
    def total_tokens(self) -> int:
        return sum(
            msg.content.tokens(enc=self.token_encoding) for msg in self.messages_
        )

    @property
    def messages(self):
        return self.messages_

    def __request_payload__(self, max_tokens: int, stream: bool) -> dict:
        tool = {
            "type": "function",
            "function": {
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
        }
        return {
            "model": self.model_name,
            "temperature": 0.3,
            "messages": self.to_dict(),
            "stream": stream,
            "max_tokens": max_tokens,
            "tools": [tool],
            "tool_choice": "auto",
        }

    def add_user_msg(self, msg: str) -> None:
        self.append(User(msg=msg))

    async def add_assistant_msg(
        self,
        content: Union[TextContent, CodeExecutionContent, FunctionCallContent] = None,
        **kwargs,  # Additional keyword arguments for Assistant (e.g., name, tool_call_id)
    ) -> None:
        """
        Adds an assistant message to the conversation with flexible content types.

        Args:
                        content: The content of the assistant's message. Can be TextContent, CodeExecutionContent, or FunctionCallContent.
                        **kwargs: Additional keyword arguments to pass to the Assistant constructor (e.g., name, tool_call_id).
        """

        if content is None:
            raise ValueError("Content must be provided.")

        assistant_message = Assistant(content=content, **kwargs)
        self.append(assistant_message)

    def append(self, message: Union[User, Assistant, System]) -> None:
        self.messages_.append(message)

    def to_dict(self) -> List[dict]:
        return [
            {
                "role": m.role,
                "content": (
                    m.content.text
                    if isinstance(m.content, TextContent)
                    else m.content.model_dump_json()
                ),
            }
            for m in self.messages_
        ]

    async def call_llm(self, max_tokens: int, stream: bool = False):
        """Calls the LLM, handles responses, and manages tool calls."""
        try:
            response = await acompletion(**self.__request_payload__(max_tokens, stream))

            if stream:
                current_tool_call = None
                async for chunk in response:
                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason

                    # If nothing is received, continue the loop
                    if delta is None:
                        continue

                    # Handle tool calls
                    if delta.tool_calls:
                        tool_call_info = delta.tool_calls[0]

                        if current_tool_call is None:
                            current_tool_call = FunctionCallContent(
                                name=tool_call_info.function.name or "",
                                arguments="",
                                tool_call_id=tool_call_info.id,
                            )

                        if tool_call_info.function.arguments:
                            current_tool_call.arguments += (
                                tool_call_info.function.arguments
                            )

                    # Handle regular content
                    elif delta.content:
                        yield delta.content, finish_reason

                    # Handle finish conditions
                    if finish_reason == "tool_calls":
                        if current_tool_call:
                            yield current_tool_call, finish_reason
                            current_tool_call = None
                    elif finish_reason:
                        if current_tool_call:
                            yield current_tool_call, "tool_calls"
                            current_tool_call = None
                        else:
                            yield None, finish_reason

            else:  # Non-streamed response
                message = response.choices[0].message
                if message.tool_calls:
                    tool_call = message.tool_calls[0]
                    yield FunctionCallContent(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                        tool_call_id=tool_call.id,
                    ), "tool_calls"
                else:
                    yield message.content, "stop"

        except Exception as e:
            logging.exception(f"Error in call_llm: {e}")
            raise
