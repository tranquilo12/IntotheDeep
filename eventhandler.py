import json
from typing import Optional

import chainlit as cl

import tools
from llm_types import (
    CodeExecutionContent,
    Conversation,
    FunctionCallContent,
    TextContent,
)
from tools import data_analyst


#############################################
############## Event Handler ################
#############################################


# noinspection DuplicatedCode
class ChainlitEventHandler:
    def __init__(self, conversation: "Conversation"):
        """
        Initialize the ChainlitEventHandler.

        Parameters
        ----------
        conversation : Conversation
            The conversation object.
        """
        self.conversation = conversation
        self.current_step = None
        self.streaming_response = None
        self.current_tool_call = None
        self.function_call_made = False
        self.initial_tokens = self.conversation.total_tokens
        self.token_text = None
        self.tokens_used = 0
        self.current_message_content = ""
        self.buffer = ""

    async def handle_sse_line(self, line: str) -> None:
        """
        Handle a single line from the SSE stream.

        Parameters
        ----------
        line : str
            A line from the SSE stream.
        """
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
        """
        Handle a chunk of data from the conversation.

        Parameters
        ----------
        chunk :
            The chunk of data to handle.
        """
        delta = chunk["choices"][0]["delta"]
        finish_reason = chunk["choices"][0].get("finish_reason")

        if "content" in delta:
            content = delta["content"]
            chunk_tokens = len(self.conversation.encoding.encode(content))
            self.tokens_used += chunk_tokens
            self.current_message_content += content
            await self.handle_content(content, finish_reason)
        elif "function_call" in delta:
            await self.handle_function_call(delta["function_call"], finish_reason)

        if finish_reason:
            await self.handle_finish(finish_reason)

        await self.update_token_usage()

    async def handle_content(self, content: str, finish_reason: Optional[str]) -> None:
        """
        Handle content from the conversation.

        Parameters
        ----------
        content : str
            The content to handle.
        finish_reason : Optional[str]
            The reason for finishing.
        """
        if self.streaming_response is None:
            self.streaming_response = cl.Step(
                type="undefined"
            )  # If it's None, then it's undefined.
            await self.streaming_response.send()

        await self.streaming_response.stream_token(content)

        if finish_reason:
            await self.conversation.add_assistant_msg(
                content=TextContent(text=self.current_message_content)
            )

    async def handle_function_call(
        self, function_call: dict, finish_reason: Optional[str]
    ) -> None:
        """
        Handle a function call from the conversation.

        Parameters
        ----------
        function_call : dict
            The function call to handle.
        finish_reason : Optional[str]
            The reason for finishing.
        """
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
        """
        Update the token usage for the conversation.
        """
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
        """
        Execute the function call.
        """
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
        """
        Handle the finish of a conversation step.

        Parameters
        ----------
        finish_reason : str
            The reason for finishing.
        """
        if finish_reason == "function_call" and self.current_tool_call:
            await self.execute_function()

        elif self.streaming_response and not self.function_call_made:
            await self.conversation.add_assistant_msg(
                content=TextContent(text=self.current_message_content)
            )

        if self.streaming_response:
            await self.streaming_response.update()

        # Reset for the next interaction
        self.streaming_response = None
        self.current_step = None
        self.token_text = None
        self.current_message_content = ""
