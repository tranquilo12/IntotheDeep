from uuid import uuid4
from litellm import acompletion
from typing import List, Union, Dict
from pydantic import BaseModel, Field


class TextContent(BaseModel):
    text: str


class ImageContent(BaseModel):
    image_url: str


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

    def update_system_message(self, msg: str):
        self.messages[0] = System(msg)

    def add_user_msg(self, msg: str):
        self.messages.append(User(msg=msg))

    def add_assistant_msg(self, msg: str):
        self.messages.append(Assistant(msg=msg))

    def append(self, message: Union[User, Assistant, System]):
        self.messages.append(message)

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
            partial_message = ""
            async for chunk in response:
                if chunk.choices:
                    if (chunk.choices[0].delta.content is not None) and (
                        len(chunk.choices[0].delta.content) > 0
                    ):
                        partial_message += chunk.choices[0].delta.content
                        yield partial_message
        else:
            yield response.choices[0].message.content
            return

    async def _continue(self, max_tokens: int):
        response = self.call_llm(max_tokens=max_tokens, stream=False)
        response = "".join([p async for p in response])
        self.messages.append(Assistant(response))


class Conversations(BaseModel):
    conversations: List[Conversation]
