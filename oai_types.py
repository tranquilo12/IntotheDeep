from uuid import uuid4
from litellm import acompletion
import base64
import tiktoken
from typing import List, Union, Dict, Literal, Optional
from pydantic import BaseModel, Field, computed_field


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


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    model_name: str = Field(default="gpt-3.5-turbo")
    messages: List[Union[User, Assistant, System]] = Field(default_factory=list)
    functions: List = Field(default=list)

    def __get_encoding__(self):
        if self.model_name is not None:
            if "gpt" in self.model_name.lower():
                return tiktoken.encoding_for_model(self.model_name)

    @computed_field(return_type=int)
    @property
    def total_tokens(self):
        return sum(
            [msg.content.tokens(enc=self.__get_encoding__()) for msg in self.messages]
        )

    def update_system_message(self, msg: str):
        self.messages[0] = System(msg)

    def append(self, message: Union[User, Assistant, System]):
        self.messages.append(message)

    def add_user_msg(self, msg: str):
        self.append(User(msg=msg))

    def add_assistant_msg(self, msg: str):
        self.append(Assistant(msg=msg))

    def to_dict(self) -> List[Dict[str, str]]:
        return [
            {"role": message.role, "content": message.content.text}
            for message in self.messages
        ]

    async def call_llm(self, max_tokens: int, stream: bool):
        response = await acompletion(
            model=self.model_name,
            messages=self.to_dict(),
            stream=stream,
            max_tokens=max_tokens,
        )

        if stream:
            async for chunk in response:
                if token := chunk.choices[0].delta.content or "":
                    yield token
        else:
            yield response.choices[0].message.content
            return

    async def _continue(self, max_tokens: int, code_only: bool = False):
        # Ensure that we get the code only in the last message
        if code_only:
            latest_message = self.messages[-1].content.text
            latest_message += "\n Please ensure that you only provide python code in your response. Please start with ```python."
            self.messages[-1].content.text = latest_message

        response = self.call_llm(max_tokens=max_tokens, stream=False)
        assistant_message = Assistant("".join([p async for p in response]))
        self.append(assistant_message)


class Conversations(BaseModel):
    conversations: List[Conversation]
