import json
import base64
import tiktoken
import logging
import asyncio
from uuid import uuid4
from litellm import acompletion
from interpreter import execute_code_locally
from pydantic import BaseModel, Field, computed_field
from typing import List, Union, Dict, Literal, Optional, Callable, Any


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
        markdown_text = f"`python\n{self.code}\n`\n\n"

        if self.stdout:
            markdown_text += f"**Output:**\n`\n{self.stdout}\n`\n\n"

        if self.stderr:
            markdown_text += f"**Error:**\n`\n{self.stderr}\n`"

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
    content: Union[
        TextContent, CodeExecutionContent
    ]  # Allow either TextContent or CodeExecutionContent
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

        if msg:  # If a regular text message is provided
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
class ExecCodeLocallyParam(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Dict[str, str]] = {
        "code": {
            "type": "string",
            "description": "The python code to be executed locally. It should be properly formatted.",
        }
    }
    required: List[str] = ["code"]


class ExecCodeLocallyFunction(BaseModel):
    name: str = "execute_code_locally"
    description: str = (
        "A function that executes the given Python code locally and returns the output (stdout) and error (stderr). "
        "It requires a 'code' parameter containing valid Python code."
    )
    parameters: ExecCodeLocallyParam = Field(default_factory=ExecCodeLocallyParam)


class ExecCodeLocallyTool(BaseModel):
    name: str = "execute_code_locally"
    description: str = (
        "A function that executes the given Python code locally and returns the output (stdout) and error (stderr)."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The python code to be executed locally. It should be properly formatted.",
            }
        },
        "required": ["code"],
    }


#############################################
##### For code interpreter functions ########
#############################################
def extract_code_from_response(response: str) -> str:
    case_1 = "```python"
    case_2 = "<code>"
    case_3 = "<python>"

    if case_1 in response:
        return response.split(case_1)[1].split("```")[0].strip()
    elif case_2 in response:
        return response.split(case_2)[1].split("</code")[0].strip()
    elif case_3 in response:
        return response.split(case_3)[1].split("</python")[0].strip()
    else:
        return response


async def gen_python_code_no_debug(
    convo: "Conversation", max_tokens: int
) -> "Conversation":
    # TODO, add a loading tag here.
    # Call the OpenAI or Claude API with the conversation
    # And add the response to the conversation
    await convo._continue(max_tokens=max_tokens, code_only=True)
    code = extract_code_from_response(convo.messages_[-1].content.text)

    # Execute the generated code, handle the response
    # And lower the max_attempts flag
    stdout, stderr = await execute_code_locally(code=code)

    latest_message = convo.messages_[-1].content.text
    convo.add_assistant_msg(
        f"The code generated the following: \n{latest_message} \n\n has generated the following output:\n {stdout} and the following error:\n {stderr}"
    )
    return convo


async def gen_python_code_with_debug(
    convo: "Conversation", max_tokens: int, max_attempts: int = 5
) -> "Conversation":

    # TODO, add a loading tag here.
    # Call the OpenAI or Claude API with the conversation
    # And add the response to the conversation
    await convo._continue(max_tokens=max_tokens, code_only=True)
    code = extract_code_from_response(convo.messages_[-1].content.text)

    # Execute the generated code, handle the response
    # And lower the max_attempts flag
    stdout, stderr = await execute_code_locally(code=code)

    if stderr != "":  # If error, enter a recursive loop to attempt to correct it
        if max_attempts > 0:
            convo.add_user_msg(
                "".join(
                    [
                        f"Please debug the above code that you have generated, it causes the following error: {stderr}. ",
                        "Take some time to think about your solution, within <thinking> tags.",
                        "Please provide the corrected code within '```python' and '```' tags. ",
                    ]
                )
            )

            # Recursively call self, to execute code locally
            convo = await gen_python_code_with_debug(
                convo=convo, max_tokens=max_tokens, max_attempts=max_attempts
            )

        else:  # After the max amount of attempts
            latest_message = convo.messages_[-1].content.text
            convo.add_assistant_msg(
                f"The code generated the message: \n{latest_message} \n\n after {max_attempts} attempts has generated the following error: \n {stderr}"
            )

    else:  # if there's no error, then just add the assistant message
        latest_message = convo.messages_[-1].content.text
        convo.add_assistant_msg(
            f"The code generated the following: \n{latest_message} \n\n has generated the following output:\n {stdout}"
        )

    return convo


#############################################
############### Convo Class #################
#############################################
class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    model_name: str = Field(default="gpt-3.5-turbo-0125")
    messages_: List[Union[User, Assistant, System]] = Field(default_factory=list)
    tools: List = Field(default=None)
    tool_choice: Dict = Field(default=None)
    pending_function_calls: Dict = Field(default={})
    pending_function_calls_queue: List = Field(default=[])
    active_function_call: str = Field(default="")

    @property
    def available_functions(self) -> Dict[str, Callable]:
        return {
            "gen_python_code_no_debug": gen_python_code_no_debug,
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
    def total_tokens(self):
        return sum(
            [msg.content.tokens(enc=self.token_encoding) for msg in self.messages_]
        )

    def __request_payload__(self, max_tokens: int, stream: bool):
        return {
            "model": self.model_name,
            "messages": self.to_dict(),
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "functions": [ExecCodeLocallyTool().model_dump()],
        }

    def update_system_message(self, msg: str):
        self.messages_[0] = System(msg)

    def append(self, message: Union[User, Assistant, System]):
        self.messages_.append(message)

    def add_user_msg(self, msg: str):
        self.append(User(msg=msg))

    async def add_assistant_msg(
        self,
        msg: str = None,
        name: str = None,
        tool_call_id: str = None,
        code: str = None,
        stdout: str = None,
        stderr: str = None,
    ):
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

    def del_message(self, index: int):
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

    async def call_llm(self, max_tokens: int, stream: bool):
        """
        Makes a call to the LLM, handling streaming and deferred function calls.

        Args:
            max_tokens: Maximum tokens for the LLM response.
            stream: Whether the response should be streamed.

        Yields:
            Tokens from the LLM response stream, or an empty string if waiting for function calls.
        """
        try:
            response = await acompletion(**self.__request_payload__(max_tokens, stream))
            if stream:
                async for chunk in response:
                    if token := chunk.choices[0].delta.content or "":
                        yield token
                    if "function_call" in chunk.choices[0].delta:
                        m_ = chunk.choices[0].delta
                        async for result in self.process_tool_calls(m_):
                            yield result

            else:
                first_response_message = response.choices[0].message
                yield first_response_message.content

            # Execute any pending function calls after the stream completes or after a non-streamed response
            for function_name, function_args in self.pending_function_calls_queue:
                function_to_call = self.available_functions[function_name]
                function_response = await function_to_call(
                    code=function_args.get("code")
                )  # Await asynmax_token function

                stdout, stderr = function_response
                await self.add_assistant_msg(
                    code=function_args.get("code"),
                    stdout=stdout,
                    stderr=stderr,
                    name=function_name,
                )

                # Fetch additional response after function call
                # second_response = await acompletion(
                #     **self.__request_payload__(max_tokens, False)
                # )
                # self.add_assistant_msg(msg=second_response.choices[0].message.content)

            # Clear pending function calls after execution
            self.pending_function_calls = {}
            self.pending_function_calls_queue = []

        except Exception as e:
            logging.exception("Error in call_llm: %s", str(e))
            raise

    async def _continue(self, max_tokens: int, code_only: bool = False):
        try:
            # Ensure that we get the code only in the last message
            if code_only:
                latest_message = self.messages_[-1].content.text
                latest_message += "\n Please ensure that you only provide python code in your response. Please start with ```python."
                self.messages_[-1].content.text = latest_message

            response = self.call_llm(max_tokens=max_tokens, stream=False)
            assistant_message = Assistant("".join([p async for p in response]))
            self.append(assistant_message)
        except Exception as e:
            logging.exception("Error in _continue: %s", str(e))
            raise
