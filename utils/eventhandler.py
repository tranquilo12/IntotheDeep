import json
from typing import Optional, Union

import aiohttp
import chainlit as cl

import utils.tools as tools
from src._types import (
    Assistant,
    CodeExecutionContent,
    Conversation,
    FunctionCallContent,
    System,
    TextContent,
    User,
)
from utils.tools import data_analyst


class ChainlitEventHandler:
    def __init__(self, conversation: "Conversation"):
        self.conversation: "Conversation" = conversation
        self.current_step: Optional[cl.Step] = None
        self.streaming_response: Optional[Union[cl.Step, cl.Message]] = None
        self.current_tool_call: Optional[FunctionCallContent] = None
        self.function_call_made: bool = False
        self.initial_tokens: int = self.conversation.total_tokens
        self.token_text: Optional[cl.Text] = None
        self.tokens_used: int = 0
        self.current_message_content: str = ""
        self.buffer: str = ""

    async def call_llm(self, max_tokens: int = 4096):
        payload = self.conversation.__payload__(max_tokens)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {payload.api_key}",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=payload.api_base + "/chat/completions",
                headers=headers,
                json=payload.model_dump(exclude={"api_key", "api_base"}),
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        await self.handle_sse_line(line.decode("utf-8").strip())
                else:
                    error_text = await response.text()
                    print(f"Error calling LLM: HTTP {response.status}, {error_text}")

    async def add_message(self, message: Assistant | User | System):
        self.conversation.messages_.append(message)
        if isinstance(message, Assistant):
            message = cl.Message(content=message.content.text, author=message.role)
            await message.send()

    async def handle_sse_line(self, line: str) -> None:
        if line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                await self.handle_finish("stop")
            else:
                try:
                    chunk = json.loads(data)
                    await self.handle_chunk(chunk)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {data}")
        else:
            self.buffer += line
            if self.buffer.endswith("\n\n"):
                try:
                    chunk = json.loads(self.buffer)
                    await self.handle_chunk(chunk)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {self.buffer}")
                self.buffer = ""

    async def handle_chunk(self, chunk) -> None:
        delta = chunk["choices"][0]["delta"]
        finish_reason = chunk["choices"][0].get("finish_reason")

        if "content" in delta:
            content = delta["content"]
            chunk_tokens = len(self.conversation.encoding.encode(content))
            self.tokens_used += chunk_tokens
            self.current_message_content += content
            await self.handle_content(content, finish_reason)
        elif "function_call" in delta:
            self.function_call_made = True
            await self.handle_function_call(delta["function_call"], finish_reason)

        if finish_reason:
            await self.handle_finish(finish_reason)

        await self.update_token_usage()

    async def handle_content(self, content: str, finish_reason: Optional[str]) -> None:
        if self.streaming_response is None:
            self.streaming_response = cl.Message(content="", author="Assistant")
            await self.streaming_response.send()

        self.streaming_response.content += content
        await self.streaming_response.update()

        if finish_reason:
            await self.add_message(
                message=Assistant(TextContent(text=self.current_message_content))
            )

    async def handle_function_call(
        self, function_call: dict, finish_reason: Optional[str]
    ) -> None:
        self.function_call_made = True

        if self.current_tool_call is None:
            self.current_tool_call = FunctionCallContent(
                name=function_call["name"],
                arguments=function_call["arguments"],
            )
            self.current_step = cl.Step(type="tool", name=self.current_tool_call.name)
            await self.current_step.send()

        if function_call["arguments"]:
            self.current_tool_call.arguments += function_call["arguments"]
            await self.current_step.stream_token(
                function_call["arguments"], is_input=True
            )

        if finish_reason == "function_call":
            await self.execute_function()

    async def update_token_usage(self):
        current_tokens = self.conversation.total_tokens

        if self.token_text is None:
            self.token_text = cl.Text(
                content=str(current_tokens),
                name="Token Count",
            )
        else:
            self.token_text.content = str(current_tokens)

        if self.streaming_response:
            if not self.streaming_response.elements:
                self.streaming_response.elements = [self.token_text]
            else:
                self.streaming_response.elements[0] = self.token_text

            await self.streaming_response.update()

        elif self.current_step:
            if not self.current_step.elements:
                self.current_step.elements = [self.token_text]
            else:
                self.current_step.elements[0] = self.token_text

            await self.current_step.update()

    async def execute_function(self):
        if self.current_tool_call.name in [
            "execute_code_locally",
            "debug_code",
            "python",
            "functions",
        ]:
            try:
                function_to_call = getattr(tools, self.current_tool_call.name, None)
                code = json.loads(self.current_tool_call.arguments)["code"]
            except ValueError as _:
                function_to_call = None
                code = self.current_tool_call.arguments

            if function_to_call:
                accumulated_stdout = ""
                async for result_chunk in function_to_call(
                    code, self.conversation.interpreter
                ):
                    if isinstance(result_chunk, CodeExecutionContent):
                        accumulated_stdout += result_chunk.stdout + "\n\n"
                        result_chunk.stdout = accumulated_stdout
                        self.current_step.output = result_chunk.text
                        await self.conversation.add_assistant_msg(content=result_chunk)
                        if _ := await self.current_step.update():
                            self.current_tool_call = None
            else:
                print(f"Function {self.current_tool_call.name} not found.")

        elif self.current_tool_call.name in [
            "generate_code",
            "data_generator",
        ]:
            try:
                function_to_call = getattr(tools, self.current_tool_call.name, None)
                instructions = json.loads(self.current_tool_call.arguments)[
                    "instructions"
                ]
            except ValueError as _:
                function_to_call = None
                instructions = self.current_tool_call.arguments

            if function_to_call:
                async for result_chunk in function_to_call(instructions):
                    if isinstance(result_chunk, CodeExecutionContent):
                        self.current_step.output = result_chunk.text
                        await self.conversation.add_assistant_msg(content=result_chunk)
                        if _ := await self.current_step.update():
                            self.current_tool_call = None
            else:
                print(f"Function {self.current_tool_call.name} not found.")

        elif self.current_tool_call.name in [
            "get_latest_changes_within_git",
            "get_git_commit_prompt",
        ]:
            try:
                function_to_call = getattr(tools, self.current_tool_call.name, None)
                root_path = json.loads(self.current_tool_call.arguments)["root_path"]
            except ValueError as _:
                function_to_call = None
                root_path = self.current_tool_call.arguments

            if function_to_call:
                async for result_chunk in function_to_call(root_path):
                    if isinstance(result_chunk, TextContent):
                        self.current_step.output = result_chunk.text
                        await self.conversation.add_assistant_msg(content=result_chunk)
                        if _ := await self.current_step.update():
                            self.current_tool_call = None
            else:
                print(f"Function {self.current_tool_call.name} not found.")
        else:
            try:
                file_path = json.loads(self.current_tool_call.arguments)["file_path"]
                instructions = json.loads(self.current_tool_call.arguments)[
                    "instructions"
                ]
            except ValueError as _:
                [file_path, instructions] = self.current_tool_call.arguments
            async for result_chunk in data_analyst(file_path, instructions):
                if isinstance(result_chunk, TextContent):
                    self.current_step.output = result_chunk.text
                    await self.conversation.add_assistant_msg(content=result_chunk)
                    if _ := await self.current_step.update():
                        self.current_tool_call = None

    async def handle_finish(self, finish_reason: str) -> None:
        if finish_reason == "function_call" and self.current_tool_call:
            await self.execute_function()

        elif self.streaming_response and not self.function_call_made:
            await self.streaming_response.update()

        if self.streaming_response:
            await self.streaming_response.update()

        # Reset for the next interaction
        self.streaming_response = None
        self.current_step = None
        self.token_text = None
        self.current_message_content = ""
