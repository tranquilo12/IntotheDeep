"""
Something cool here
"""

import json
from typing import Dict

import chainlit as cl
from dotenv import load_dotenv

from models import ModelNames
from oai_types import (
    ChainlitEventHandler,
    CodeExecutionContent,
    Conversation,
    FunctionCallContent,
    execute_code_locally,
)
from utils import init_convo

load_dotenv()

# Global variable to track token count across sessions (optional)
TOTAL_TOKENS: int = 0


#############################################
########## Helper Functions (Convo) #########
#############################################
@cl.step(type="tool", show_input=False)
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
    # current_step.language = "json"

    # Accumulate arguments
    tool_call_id = tool_call.tool_call_id
    accumulated_args = cl.user_session.get(f"accumulated_args_{tool_call_id}", "")
    accumulated_args += tool_call.arguments
    await current_step.stream_token(token=tool_call.arguments, is_input=True)

    CONVO = cl.user_session.get("CONVO")
    if finish_reason == "tool_call":
        cl.user_session.set(f"accumulated_args_{tool_call_id}", None)  # Clear arguments
        try:
            args = json.loads(accumulated_args)
        except json.decoder.JSONDecodeError as _:
            # check if there's ```python tags and then execute the code.
            code = accumulated_args.lstrip("```python").rstrip("```")
            args = {"code": code}

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
    CONVO: Conversation = cl.user_session.get("CONVO")
    BEFORE: int = CONVO.total_tokens
    tokens_used = cl.Text(
        name="Tokens Used", content=f"Tokens Used: {str(BEFORE)}", display="inline"
    )

    event_handler = ChainlitEventHandler(CONVO)
    await CONVO.call_llm(max_tokens=max_tokens, event_handler=event_handler)

    AFTER: int = CONVO.total_tokens
    tokens_gen = cl.Text(
        name="Tokens Generated",
        content=f"Tokens Generated: {str(AFTER - BEFORE)}",
        display="inline",
    )

    if event_handler.streaming_response:
        event_handler.streaming_response.elements = [tokens_used, tokens_gen]
        await event_handler.streaming_response.update()

    cl.user_session.set("CONVO", CONVO)


@cl.on_chat_start
async def on_chat_start():
    """
    All things that happen when you're about to start convo.
    """
    # Ask the user to select a model before initializing the conversation
    model_selection = await cl.AskActionMessage(
        content="Please select a model to use for this conversation:",
        actions=[
            cl.Action(name="model", value=model.value, label=model.value)  # type: ignore
            for model in ModelNames
        ],
    ).send()

    if model_selection is None:
        await cl.Message(content="No model selected. Using default model.").send()
        selected_model = ModelNames.GPT_3_5_TURBO.value
    else:
        selected_model = model_selection["value"]

    # Now we can use the selected model when initializing the conversation
    first_msg: cl.types.StepDict | None = await cl.AskUserMessage(
        content="What do you want to do?",
        type="assistant_message",
        timeout=60,
    ).send()

    if first_msg:
        CONVO: Conversation = init_convo(
            context_code="No code provided, ignore.",
            user_question=first_msg["output"],
            model_name=selected_model,  # Use the selected model here
        )

        # Cache the updated conversation
        cl.user_session.set("CONVO", CONVO)

        # Let's start the conversation
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
    CONVO.model = settings["Model"]

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
