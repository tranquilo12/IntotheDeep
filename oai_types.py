import base64
import json
from typing import Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4

import aiohttp
import chainlit as cl
import tiktoken
from litellm import acompletion
from litellm.types.utils import Delta, ModelResponse
from openai.types.chat.chat_completion_message import ChatCompletionMessageToolCall
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
        markdown_text = f"**Generated Code:**\n---\n```python\n{self.code} \n```\n\n"

        if self.stdout:
            markdown_text += f"**Output:**\n---\n```stdout\n{self.stdout} \n``` \n\n"

        if self.stderr:
            markdown_text += f"**Error:**\n---\n```stderr\n{self.stderr} \n``` \n\n"

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
    type: Literal["tool_call"] = "tool_call"
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
########## Chainlit Event handler ###########
#############################################
class ChainlitEventHandler:
    def __init__(self, conversation: "Conversation"):
        self.conversation: Conversation = conversation
        self.current_step: Optional[cl.Step] = None
        self.streaming_response: Optional[cl.Message] = None
        self.current_tool_call = None
        self.function_call_made = None

    async def handle_chunk(self, chunk: ModelResponse):
        delta: Delta = chunk.choices[0].delta
        finish_reason: Optional[str] = chunk.choices[0].finish_reason

        if delta.content is not None:
            await self.handle_content(delta.content, finish_reason)
        elif delta.tool_calls:
            # We're just sending one tool here (execute_code_locally)
            # TODO: Ensure this is made into a parallel function call later.
            if finish_reason is not None:
                print(f"{finish_reason=}")

            await self.handle_tool_call(delta.tool_calls[0], finish_reason)

        if finish_reason:
            await self.handle_finish(finish_reason)

    async def handle_content(self, content: str, finish_reason: Optional[str]):
        if self.streaming_response is None:
            self.streaming_response = cl.Message("")
            await self.streaming_response.send()
        await self.streaming_response.stream_token(content)
        if finish_reason:
            await self.conversation.add_assistant_msg(
                content=TextContent(text=self.streaming_response.content)
            )

    async def handle_tool_call(
        self, tool_call: ChatCompletionMessageToolCall, finish_reason: Optional[str]
    ):
        self.function_call_made = True

        if self.current_tool_call is None:
            self.current_tool_call = FunctionCallContent(
                name=tool_call.function.name, arguments=tool_call.function.arguments
            )
            self.current_step = cl.Step(type="tool", name=self.current_tool_call.name)
            await self.current_step.send()

        if tool_call.function.arguments:
            self.current_tool_call.arguments += tool_call.function.arguments
            await self.current_step.stream_token(
                tool_call.function.arguments, is_input=True
            )

        if finish_reason:
            await self.execute_function()

    async def execute_function(self):
        try:
            code = json.loads(self.current_tool_call.arguments)["code"]
        except ValueError as _:
            code = self.current_tool_call.arguments

        if self.current_tool_call.name in [
            "execute_code_locally",
            "python",
            "functions",
        ]:
            async for result_chunk in execute_code_locally(
                code, self.conversation.interpreter
            ):
                if isinstance(result_chunk, CodeExecutionContent):
                    self.current_step.output = result_chunk.text
                    await self.conversation.add_assistant_msg(content=result_chunk)

        if _ := await self.current_step.update():
            self.current_tool_call = None

    async def handle_finish(self, finish_reason: str):
        if finish_reason and self.current_tool_call:
            await self.execute_function()
        elif self.streaming_response and not self.function_call_made:
            await self.conversation.add_assistant_msg(
                content=TextContent(text=self.streaming_response.content)
            )
        if self.streaming_response:
            await self.streaming_response.update()


#############################################
############### Convo Class #################
#############################################
class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    model: str = Field(default=ModelNames.GPT_3_5_TURBO.value)
    accumulated_arguments: Dict = Field(default={})
    active_tool_calls: Dict = Field(default={})
    messages_: List[Union[User, Assistant, System]] = Field(default_factory=list)
    interpreter: Interpreter = Field(default_factory=Interpreter)

    @property
    def encoding(self) -> tiktoken.Encoding:
        if self.model is not None:
            if "gpt" in self.model.lower():
                return tiktoken.encoding_for_model(self.model)

    @property
    def total_tokens(self) -> int:
        return sum(msg.content.tokens(enc=self.encoding) for msg in self.messages_)

    @property
    def messages(self):
        return self.messages_

    def __append__(self, message: Union[User, Assistant, System]) -> None:
        self.messages_.append(message)

    def __to_dict__(self) -> List[dict]:
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

    def __payload__(self, max_tokens: int) -> dict:
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
            "model": self.model,
            "temperature": 0.3,
            "messages": self.__to_dict__(),
            "stream": True,
            "max_tokens": max_tokens,
            "tools": [tool],
            "tool_choice": "auto",
        }

    def add_user_msg(self, msg: str) -> None:
        self.__append__(User(msg=msg))

    async def add_assistant_msg(
        self,
        content: Optional[
            Union[TextContent, CodeExecutionContent, FunctionCallContent]
        ] = None,
        **kwargs,  # (e.g., name, tool_call_id)
    ) -> None:
        """
        Adds an assistant message to the conversation with flexible content types.

        Parameters:
        -----------
        content: The content of the assistant's message. Can be TextContent, CodeExecutionContent, or FunctionCallContent.
        **kwargs: Additional keyword arguments to pass to the Assistant constructor (e.g., name, tool_call_id).
        """
        if content is None:
            raise ValueError("Content must be provided.")

        assistant_message = Assistant(content=content, **kwargs)
        self.__append__(assistant_message)

    async def call_llm(self, max_tokens: int, event_handler: ChainlitEventHandler):
        """Calls the LLM, handles responses, and manages tool calls."""
        payload = self.__payload__(max_tokens=max_tokens)
        response = await acompletion(**payload)
        async for chunk in response:
            await event_handler.handle_chunk(chunk)
