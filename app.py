"""
Something cool here
"""

import json
from typing import Dict

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
@cl.step(type="tool", show_input=True)
async def call_tool(tool_call: FunctionCallContent, finish_reason: str):
    """
    Just for calling tools...

    Parameters
    ----------
    tool_call : FunctionCallContent, The content from the
    finish_reason : str,
    """
    current_step = cl.context.current_step
    function_name = tool_call.name
    current_step.name = function_name
    current_step.input = None

    # Accumulate arguments
    tool_call_id = tool_call.tool_call_id
    accumulated_args = cl.user_session.get(f"accumulated_args_{tool_call_id}", "")
    accumulated_args += tool_call.arguments
    await current_step.stream_token(token=tool_call.arguments, is_input=True)

    CONVO = cl.user_session.get("CONVO")
    if finish_reason == "tool_calls":
        cl.user_session.set(f"accumulated_args_{tool_call_id}", None)  # Clear arguments
        args = json.loads(accumulated_args)

        # Execute the tool and stream results
        if function_name == "execute_code_locally":
            async for result_chunk in execute_code_locally(
                args["code"], CONVO.interpreter
            ):
                if isinstance(result_chunk, CodeExecutionContent):
                    current_step.output = result_chunk.text
                    await CONVO.add_assistant_msg(content=result_chunk)
        else:
            raise NotImplementedError(f"Function {function_name} not implemented.")

    await current_step.send()


# noinspection PyArgumentList
async def run_conversation(max_tokens: int = 4000):
    """
    Run the convo! Send to the LLM!

    Parameters
    ----------
    max_tokens : int, It's always the max.
    """
    global TOTAL_TOKENS
    CONVO: Conversation = cl.user_session.get("CONVO")

    while True:
        BEFORE = CONVO.total_tokens
        tokens_used = cl.Text(content=str(BEFORE), name="Tokens used")
        llm_response = CONVO.call_llm(max_tokens=max_tokens, stream=True)

        streaming_response = cl.Message(content="", elements=[tokens_used])
        await streaming_response.send()

        complete_str_resp = ""
        async for chunk, finish_reason in llm_response:
            if isinstance(chunk, FunctionCallContent):
                await call_tool(chunk, finish_reason)

            elif isinstance(chunk, str):
                await streaming_response.stream_token(chunk)
                complete_str_resp += chunk

                if finish_reason is not None:
                    content = TextContent(text=complete_str_resp)
                    await CONVO.add_assistant_msg(content=content)

        cl.user_session.set("CONVO", CONVO)

        AFTER = CONVO.total_tokens
        tokens_gen = cl.Text(content=str(AFTER - BEFORE), name="Tokens Generated")
        streaming_response.elements = [tokens_used, tokens_gen]
        await streaming_response.update()
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

    # files = None
    # while files is None:
    #     files: List[cl.types.AskFileResponse] | None = await cl.AskFileMessage(
    #         content="Please upload python files only.",
    #         accept={
    #             "text/plain": [".txt", ".py", ".env", ".html", ".css", ".js", ".csv"]
    #         },
    #         max_size_mb=10,
    #         timeout=240,
    #         max_files=4,
    #     ).send()
    #
    # all_code = []
    # for py_f in files:
    #     with open(py_f.path, "r", encoding="utf-8") as f:
    #         code = f.read()
    #         formatted_code = f"### filename: {py_f.name} ###\n\n{code}\n\n###"
    #         all_code.append(formatted_code)

    first_msg: cl.types.StepDict | None = await cl.AskUserMessage(
        content="What do you want to do with these uploaded files?",
        type="assistant_message",
        timeout=60,
    ).send()

    if first_msg:
        CONVO: Conversation = init_convo(
            # context_code="\n\n".join(all_code),
            context_code="No code provided, ignore.",
            user_question=first_msg["output"],
            model_name=settings["Model"],
        )

        # Cache the updated conversation, before you probe_llm, cause async magic
        cl.user_session.set("CONVO", CONVO)

        # Let's goooo
        await run_conversation()


@cl.on_settings_update
async def on_settings_update(settings: Dict):
    """
    For all settings update

    Parameters
    ----------
    settings : Dict
    """
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
