import asyncio
import base64
import json
import logging
from typing import List, Union, Dict, Literal, Optional, Callable, Any
from uuid import uuid4

import tiktoken
from litellm import acompletion
from pydantic import BaseModel, Field, computed_field

from interpreter import execute_code_locally


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
        markdown_text = f"```python\n{self.code}\n```\n"

        if self.stdout:
            markdown_text += f"**Output:**\n```\n{self.stdout}\n```\n\n"

        if self.stderr:
            markdown_text += f"**Error:**\n```\n{self.stderr}\n```"

        return markdown_text

    def tokens(self, enc: tiktoken.Encoding) -> int:
        # Use the generated markdown text for token counting
        return len(enc.encode(self.text))


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

    def tokens(self, enc: tiktoken.Encoding) -> int:
        return len(enc.encode(self.text))


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: Optional[str | Dict[str, str]]

    @classmethod
    def validate_image_path(cls, image_url):
        with open(image_url, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_url = {"url": f"data:image/jpeg;base64,{encoded_image}"}
        return image_url

    def tokens(self, enc: tiktoken.Encoding) -> int:
        return len(enc.encode(self.image_url))


class User(BaseModel):
    role: str = "user"
    content: TextContent

    def __init__(self, msg: str, **data):
        super().__init__(content=TextContent(text=msg), **data)


class Assistant(BaseModel):
    role: str = "assistant"
    content: Union[TextContent, CodeExecutionContent]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def __init__(
            self,
            msg: str = None,
            name: str = None,
            tool_call_id: str = None,
            code: str = None,
            stdout: str = None,
            stderr: str = None,
            **data,
    ):

        if msg is not None:  # If a regular text message is provided
            content = TextContent(text=msg)
        elif code is not None and stdout is not None:  # If code and output are provided
            content = CodeExecutionContent(code=code, stdout=stdout, stderr=stderr)
        else:
            raise ValueError("Either msg or (code and stdout) must be provided")

        super().__init__(content=content, name=name, tool_call_id=tool_call_id, **data)


class System(BaseModel):
    role: str = "system"
    content: TextContent

    def __init__(self, msg: str, **data):
        super().__init__(content=TextContent(text=msg), **data)


#############################################
############### For all tools ###############
#############################################
class Parameter(BaseModel):
    type: str
    description: str


class CodeBlockParameter(Parameter):
    type: Literal["string"] = "string"
    description: str = "The Python code to execute, enclosed in triple backticks (```)."


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []},
        description="The parameters for the function call.",
    )


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ExecCodeLocallyTool(Tool):
    function: FunctionDefinition = FunctionDefinition(
        name="execute_code_locally",
        description="Execute provided Python code locally and return the standard output and standard error.",
    )

    def add_parameter(self, name: str, parameter: Parameter, required: bool = False):
        """Adds a parameter to the function definition."""
        self.function.parameters["properties"][name] = parameter.model_dump(
            exclude_none=True
        )
        if required:
            self.function.parameters["required"].append(name)


#############################################
############### Convo Class #################
#############################################
class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    model_name: str = Field(default="gpt-3.5-turbo-0125")
    messages_: List[Union[User, Assistant, System]] = Field(default_factory=list)
    tools: List = Field(default=None)
    tool_choice: Dict = Field(default=None)
    accumulated_code_: str = Field(default="")
    pending_function_calls: Dict = Field(default={})
    pending_function_calls_queue: List = Field(default=[])
    active_function_call: str = Field(default="")

    @property
    def available_functions(self) -> Dict[str, Callable]:
        return {
            "execute_code_locally": execute_code_locally,
        }

    @property
    def token_encoding(self) -> tiktoken.Encoding:
        if self.model_name is not None:
            if "gpt" in self.model_name.lower():
                return tiktoken.encoding_for_model(self.model_name)

    @property
    def messages(self):
        return self.messages_

    @messages.setter
    def messages(self, value):
        raise AttributeError(
            "Cannot set messages directly. Use append() method instead."
        )

    @computed_field(return_type=int)
    @property
    def total_tokens(self) -> int:
        return sum(
            [msg.content.tokens(enc=self.token_encoding) for msg in self.messages_]
        )

    def __request_payload__(self, max_tokens: int, stream: bool) -> Dict:
        tool = ExecCodeLocallyTool()
        tool.add_parameter("code", CodeBlockParameter(), required=True)
        return {
            "model": self.model_name,
            "temperature": 0.3,
            "messages": self.to_dict(),
            "stream": stream,
            "max_tokens": max_tokens,
            "tools": [tool.model_dump(exclude_none=True)],
            "tool_choice": "auto",
        }

    def update_system_message(self, msg: str) -> None:
        self.messages_[0] = System(msg)

    def append(self, message: Union[User, Assistant, System]) -> None:
        self.messages_.append(message)

    def add_user_msg(self, msg: str) -> None:
        self.append(User(msg=msg))

    async def add_assistant_msg(
            self,
            msg: str = None,
            name: str = None,
            tool_call_id: str = None,
            code: str = None,
            stdout: str = None,
            stderr: str = None,
    ) -> None:
        """Asynchronously adds an assistant message to the conversation."""
        if msg is not None:
            assistant_message = Assistant(msg=msg, name=name, tool_call_id=tool_call_id)
        elif code is not None and stdout is not None:
            if isinstance(stdout, asyncio.Future) or isinstance(stderr, asyncio.Future):
                # If stdout or stderr are futures, await them
                stdout, stderr = await asyncio.gather(stdout, stderr)

            assistant_message = Assistant(
                code=code,
                stdout=stdout,
                stderr=stderr,
                name=name,
                tool_call_id=tool_call_id,
            )
        else:
            raise ValueError("Either msg or (code, stdout) must be provided")

        self.append(assistant_message)  # Add the assistant message to the conversation

    def del_message(self, index: int) -> None:
        if 0 <= index < len(self.messages_):
            del self.messages_[index]
        else:
            raise IndexError("Invalid message index.")

    def to_dict(self) -> List[Dict[str, str]]:
        return [
            {"role": message.role, "content": message.content.text}
            for message in self.messages_
        ]

    async def process_tool_calls(self, message):
        """
        Handles function calls received in the streamed LLM response, potentially in multiple chunks.

        This function now accumulates function call arguments and remembers the
        function name even if it's not repeated in subsequent chunks.

        Args:
            message: The message chunk from the LLM stream.
            max_tokens: Maximum tokens for any subsequent LLM calls after function execution.

        Yields:
            An empty string to maintain the stream flow.
        """

        function_call = message.get("function_call")
        if function_call:
            function_name = function_call.name
            function_args = function_call.arguments

            # Track the active function call
            if function_name:  # Update active function if provided
                self.active_function_call = function_name
            else:
                function_name = self.active_function_call  # Use the remembered name

            if function_name not in self.pending_function_calls:
                self.pending_function_calls[function_name] = ""

            self.pending_function_calls[function_name] += function_args

            # If the function call arguments are complete, move them to the queue
            if function_args.endswith("}"):  # Assuming JSON format for arguments
                args = self.pending_function_calls.pop(
                    function_name
                )  # Remove from pending_function_calls
                try:
                    parsed_args = json.loads(args)
                except json.JSONDecodeError as e:
                    print(f"Error parsing function arguments: {e}")
                    parsed_args = {}

                self.pending_function_calls_queue.append((function_name, parsed_args))

        yield ""  # Continue the stream without waiting for the function result

    async def execute_tool_calls(self, tool_calls=None):
        """
        Executes tool calls and updates the conversation, accumulating arguments across chunks if necessary.

        Parameters
        ----------
        tool_calls : list or None, optional
            A list of tool calls received from the LLM. If None, it indicates that there are no more tool calls to process.
        """
        if tool_calls is None:
            return  # Handle the case where there are no tool calls

        for tool_call in tool_calls:
            if self.active_function_call == "":
                if tool_call.function.name is not None:
                    self.active_function_call = tool_call.function.name
                elif tool_call.name is not None:
                    self.active_function_call = tool_call['name']

            function_name = self.active_function_call

            if function_name == "execute_code_locally":
                # Accumulate arguments if streaming and "code" not directly present
                if not hasattr(self, "accumulated_code_"):
                    self.accumulated_code_ = ""

                function_args = tool_call.function.arguments if hasattr(tool_call, "function") else tool_call.get(
                    "arguments")
                self.accumulated_code_ += function_args

                try:
                    # Attempt to parse the accumulated code as JSON
                    function_args = json.loads(self.accumulated_code_)
                except json.JSONDecodeError:
                    # If it's not valid JSON, assume it's a partial code snippet
                    pass
                else:  # If parsing was successful
                    del self.accumulated_code_  # Reset for the next code block

                    function_to_call = self.available_functions[function_name]
                    function_response = await function_to_call(**function_args)

                    stdout, stderr = function_response
                    await self.add_assistant_msg(
                        code=function_args.get("code"),
                        stdout=stdout,
                        stderr=stderr,
                        name=function_name,
                    )

                    # Yield after executing, to allow the client to update the UI
                    yield self.messages_[-1].content.text
            else:
                # For other functions, execute immediately (assuming arguments are complete)
                function_args = tool_call.function.arguments if hasattr(tool_call, "function") else tool_call.get(
                    "arguments")
                function_args = json.loads(function_args)  # Assuming JSON arguments
                function_response = await self.available_functions[function_name](**function_args)

                await self.add_assistant_msg(
                    msg=function_response,
                    name=function_name
                )

                yield

    async def call_llm(self, max_tokens: int, stream: bool):
        """
        Makes an asynchronous call to the Language Model (LLM) and handles both streaming and function calls.

        Parameters
        ----------
        max_tokens : int
            The maximum number of tokens allowed in the LLM's response.
        stream : bool
            If True, the response will be streamed token by token; otherwise, it will be returned as a single response.

        Yields
        ------
        str or dict
            If `stream` is True, yields each token of the response as it's generated or a dictionary containing a function call.
            If `stream` is False, yields the complete response as a single string or a dictionary containing a function call.

        Raises
        ------
        Exception
            If there's an error during the LLM call.
        """
        try:
            response = await acompletion(**self.__request_payload__(max_tokens, stream))

            if stream:
                async for chunk in response:
                    yield chunk.choices[0].delta  # Yield the raw chunk for processing in `probe_llm`
            else:
                yield response.choices[0].message  # Yield the whole message for non-streaming responses
        except Exception as e:
            logging.exception("Error in call_llm: %s", str(e))
            raise e
