import json
from typing import List

import chainlit as cl
from chainlit.input_widget import Select
from dotenv import load_dotenv

from models import ModelNames
from oai_types import (
    CodeExecutionContent,
    Conversation,
    FunctionCallContent,
    TextContent,
    execute_code_locally,
)
from utils import init_convo

load_dotenv()

# Global variable to track token count across sessions (optional)
TOTAL_TOKENS: int = 0


#############################################
########## Helper Functions (Convo) #########
#############################################
@cl.step(type="tool", language="json")
async def call_tool(tool_call: FunctionCallContent, finish_reason: str):
    current_step = cl.context.current_step
    function_name = tool_call.name
    current_step.name = function_name
    current_step.streaming = True
    CONVO = cl.user_session.get("CONVO")

    # Accumulate arguments
    tool_call_id = tool_call.tool_call_id
    accumulated_args = cl.user_session.get(f"accumulated_args_{tool_call_id}", "")
    accumulated_args += tool_call.arguments

    if finish_reason == "function_call":
        cl.user_session.set(f"accumulated_args_{tool_call_id}", None)  # Clear arguments
        args = json.loads(accumulated_args)

        # Execute the tool and stream results directly
        if function_name == "execute_code_locally":
            current_step.input = accumulated_args
            streaming_response = cl.Message(
                content=""
            )  # Create a message for streaming
            await streaming_response.send()
            final_response = ""
            async for result_chunk in execute_code_locally(
                args["code"], CONVO.interpreter
            ):
                if isinstance(result_chunk, CodeExecutionContent):
                    await streaming_response.stream_token(result_chunk.text)
                    final_response += result_chunk.text
            await CONVO.add_assistant_msg(
                content=CodeExecutionContent(text=final_response)
            )  # Update conversation with final response
            await streaming_response.send()  # Send the complete message
        else:
            raise NotImplementedError(f"Function {function_name} not implemented.")


# noinspection PyArgumentList
async def run_conversation(max_tokens: int = 4000):
    """Probes the LLM for a response and handles it, including code execution."""
    global TOTAL_TOKENS  # Keep track of total tokens
    CONVO: Conversation = cl.user_session.get("CONVO")

    while True:
        TOTAL_TOKENS = CONVO.total_tokens  # Update token count before the LLM call
        tokens_used = cl.Text(content=str(TOTAL_TOKENS), name="Tokens used")

        # Get the LLM's response (streamed or non-streamed)
        llm_response = CONVO.call_llm(max_tokens=max_tokens, stream=True)
        if isinstance(llm_response, str):  # Handle non-streamed content
            await CONVO.add_assistant_msg(TextContent(text=llm_response))
            normal_response = cl.Message(content=llm_response, elements=[tokens_used])
            await normal_response.send()

        else:  # Handle the streaming response
            streaming_response = cl.Message(content="", elements=[tokens_used])
            await streaming_response.send()

            final_response = ""
            async for chunk, finish_reason in llm_response:
                if isinstance(chunk, FunctionCallContent):  # Handling func calls
                    await call_tool(chunk, finish_reason)

                elif isinstance(chunk, str):  # Handle regular content
                    await streaming_response.stream_token(chunk)
                    final_response += chunk
                    await CONVO.add_assistant_msg(
                        content=TextContent(text=final_response)
                    )

            await streaming_response.send()

        cl.user_session.set("CONVO", CONVO)
        break


@cl.on_chat_start
async def on_chat_start():
    """
    ALl things that happen, when you're about to start convo.

    Returns
    -------
    None
    """
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=ModelNames.all_to_list(),
                initial_value=ModelNames.GPT_3_5_TURBO.value,
            ),
        ]
    ).send()

    files = None
    while files is None:
        files: List[cl.types.AskFileResponse] | None = await cl.AskFileMessage(
            content="Please upload python files only.",
            accept={
                "text/plain": [".txt", ".py", ".env", ".html", ".css", ".js", ".csv"]
            },
            max_size_mb=10,
            timeout=240,
            max_files=4,
        ).send()

    all_code = []
    for py_f in files:
        with open(py_f.path, "r", encoding="utf-8") as f:
            code = f.read()
            formatted_code = f"### filename: {py_f.name} ###\n\n{code}\n\n###"
            all_code.append(formatted_code)

    first_msg: cl.types.StepDict | None = await cl.AskUserMessage(
        content="What do you want to do with these uploaded files?",
        type="assistant_message",
        timeout=60,
    ).send()

    if first_msg:
        CONVO: Conversation = init_convo(
            context_code="\n\n".join(all_code),
            user_question=first_msg["output"],
            model_name=settings["Model"],
        )

        # Cache the updated conversation, before you probe_llm, cause async magic
        cl.user_session.set("CONVO", CONVO)

        # Let's goooo
        await run_conversation()


@cl.on_settings_update
async def on_settings_update(settings):
    # Get cached object
    CONVO: Conversation = cl.user_session.get("CONVO")

    # Update the conversation's model name
    CONVO.model_name = settings["Model"]

    # Cache the updated conversation, before you probe_llm, cause async magic
    cl.user_session.set("CONVO", CONVO)


@cl.on_message
async def on_message(message: cl.Message):
    """
    What's going to happen AFTER you've started the convo, and will send a message now.

    Parameters
    ----------
    message : cl.Message, Being received from the user.

    Returns
    -------
    None
    """
    # Get cached object
    CONVO: Conversation = cl.user_session.get("CONVO")

    # Ingest user message
    CONVO.add_user_msg(msg=message.content)

    # Cache the updated conversation, before you probe_llm, cause async magic
    cl.user_session.set("CONVO", CONVO)

    # Probe llm
    await run_conversation()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
