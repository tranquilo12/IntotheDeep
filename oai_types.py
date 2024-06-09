import json
import base64
import tiktoken
import logging
from uuid import uuid4
from litellm import acompletion
from interpreter import execute_code_locally
from typing import List, Union, Dict, Literal, Optional, Callable
from pydantic import BaseModel, Field, computed_field


#############################################
########## LLM related types ################
#############################################
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
    content: TextContent

    def __init__(self, msg: str, **data):
        super().__init__(content=TextContent(text=msg), **data)


class System(BaseModel):
    role: str = "system"
    content: TextContent

    def __init__(self, msg: str, **data):
        super().__init__(content=TextContent(text=msg), **data)


#############################################
############### For all tools ###############
#############################################
class CodeParameter(BaseModel):
    code: str = Field(
        ...,
        description="The valid python code, extracted from the response, to execute locally.",
    )


class ExecCodeLocally(BaseModel):
    name: str = Field(default="execute_code_locally")
    description: str = Field(
        default="A function that takes in a parameter 'code' which contains valid python code, returns (stdout, stderr)."
    )
    parameters: CodeParameter = Field(default=None)


class ExecCodeLocallyTool(BaseModel):
    type: Literal["function"] = Field(default="function")
    function: ExecCodeLocally = Field(default=None)


#############################################
############ For all tool choices ###########
#############################################
class ToolFunction(BaseModel):
    name: Literal["gen_python_code_no_debug", "execute_code_locally"]


class ExecCodeLocallyToolChoice(BaseModel):
    type: Literal["function"] = Field(default="function")
    function: ToolFunction = Field(default=ToolFunction(name="execute_code_locally"))


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
    model_name: str = Field(default="gpt-3.5-turbo")
    messages_: List[Union[User, Assistant, System]] = Field(default_factory=list)
    tools: List = Field(default=None)
    tool_choice: Dict = Field(default=None)

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
            "tools": self.tools,
            "tool_choice": self.tool_choice,
        }

    def update_system_message(self, msg: str):
        self.messages_[0] = System(msg)

    def append(self, message: Union[User, Assistant, System]):
        self.messages_.append(message)

    def add_user_msg(self, msg: str):
        self.append(User(msg=msg))

    def add_assistant_msg(self, msg: str):
        self.append(Assistant(msg=msg))

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

    async def process_tool_calls(self, message, max_tokens):
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                function_to_call = self.available_func[function_name]
                function_args = json.loads(tool_call["function"]["arguments"])
                function_response = function_to_call(code=function_args.get("code"))
                self.messages_.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            second_response = await acompletion(
                model=self.model_name,
                messages=self.to_dict(),
                stream=False,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            yield second_response.choices[0].message.content

    async def call_llm(self, max_tokens: int, stream: bool):
        try:
            request_payload = self.__request_payload__(max_tokens, stream)
            response = await acompletion(**request_payload)

            if stream:
                async for chunk in response:
                    if token := chunk.choices[0].delta.content or "":
                        yield token
                        if "tool_calls" in chunk.choices[0].delta:
                            await self.process_tool_calls(
                                chunk.choices[0].delta, max_tokens
                            )
            else:
                first_response_message = response.choices[0].message
                yield first_response_message.content
                await self.process_tool_calls(first_response_message, max_tokens)
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
