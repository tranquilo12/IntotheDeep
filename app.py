import json

import chainlit as cl
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


@cl.step(type="tool")
async def tool_call(tool_call: FunctionCallContent):
    """Executes the tool call after accumulating all arguments."""

    tool_call_id = (
        tool_call.tool_call_id
    )  # Get the unique identifier for this tool call
    function_name = tool_call.name  # Get the name of the function to be called

    # Retrieve any previously accumulated arguments for this tool call from the user session
    accumulated_args = cl.user_session.get(f"accumulated_args_{tool_call_id}", "")

    # Add the new chunk of arguments to the accumulated arguments
    accumulated_args += tool_call.arguments

    # Store the updated accumulated arguments back into the user session
    cl.user_session.set(f"accumulated_args_{tool_call_id}", accumulated_args)

    # Try to parse the accumulated arguments as a JSON object
    try:
        args = json.loads(accumulated_args)
    except json.JSONDecodeError:
        # If the arguments are not yet a valid JSON object, return (wait for more chunks)
        return

    # If the arguments are a valid JSON object, delete the accumulated arguments from the user session
    cl.user_session.set(f"accumulated_args_{tool_call_id}", None)

    # Execute the specific tool based on its name
    if function_name == "execute_code_locally":
        # If the function is to execute code locally, extract the code from the arguments
        code = args["code"]
        # Get the current conversation from the user session
        CONVO: Conversation = cl.user_session.get("CONVO")
        # Execute the code locally and get the result
        result: CodeExecutionContent = await execute_code_locally(
            code, CONVO.interpreter
        )
        # Yield the result (send it back to the main loop for further processing/display)
        return result
    else:
        # If the function name is not recognized, raise an error
        raise NotImplementedError(f"Function {function_name} not implemented.")


async def run_conversation(max_tokens: int = 4000):
    """Probes the LLM for a response and handles it, including code execution."""
    global TOTAL_TOKENS  # Keep track of total tokens
    CONVO: Conversation = cl.user_session.get("CONVO")

    while True:
        TOTAL_TOKENS = CONVO.total_tokens  # Update token count before the LLM call
        token_count_message = cl.Message(content=f"Total tokens used: {TOTAL_TOKENS}")
        await token_count_message.send()

        # Get the LLM's response (streamed or non-streamed)
        response = CONVO.call_llm(max_tokens=max_tokens, stream=True)
        if isinstance(response, str):  # Handle non-streamed content
            await CONVO.add_assistant_msg(TextContent(text=response))
            normal_response = cl.Message(content=response)
            await normal_response.send()
        else:  # Handle the streaming response
            streaming_response = cl.Message(content="")
            await streaming_response.send()
            async for chunk in response:
                if isinstance(chunk, FunctionCallContent):
                    msg = await tool_call(chunk)  # Await the final result
                    if msg:  # Check if the tool call is complete
                        await CONVO.add_assistant_msg(content=msg)
                        await streaming_response.stream_token(msg.text)
                elif content := chunk.get("content"):  # Handle regular content
                    await CONVO.add_assistant_msg(content=content)
                    await streaming_response.stream_token(content)
            await streaming_response.update()
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

    files = None
    while files is None:
        files = await cl.AskFileMessage(
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
            model_name=ModelNames.GPT_3_5_TURBO.value,
        )

        # Cache the updated conversation, before you probe_llm, cause async magic
        cl.user_session.set("CONVO", CONVO)

        # Let's goooo
        await run_conversation()


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
