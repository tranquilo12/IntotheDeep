import chainlit as cl
from dotenv import load_dotenv
from litellm.utils import Delta

from models import ModelNames
from oai_types import Conversation
from utils import init_convo

load_dotenv()

# Global variable to track token count across sessions (optional)
TOTAL_TOKENS: int = 0


async def probe_llm(max_tokens: int = 4000, from_user: bool = False):
    """
    Probes the LLM for a response to the user's message and handles the response, including code execution and tool calls.

    Parameters
    ----------
    max_tokens : int, optional
        The maximum number of tokens allowed in the LLM's response. Defaults to 500.

    from_user: bool, False
        If the message has been initiated from the user, then we're just going to ignore sending a "user message"
    """
    global TOTAL_TOKENS  # Reference global token counter
    CONVO: Conversation = cl.user_session.get("CONVO")  # Get cached object

    # Send user message immediately, if it's not sent from the user.
    # No need to render it twice
    if not from_user:
        user_message = cl.Message(content=CONVO.messages_[-1].content.text)
        await user_message.send()

    # Get response from LLM (streamed or not)
    response = CONVO.call_llm(max_tokens=max_tokens, stream=True)

    if isinstance(response, str):
        # Non-streamed response
        await CONVO.add_assistant_msg(msg=response)
        await cl.Message(content=response).send()
    else:  # Streamed response
        streaming_response = cl.Message(content="")
        await streaming_response.send()

        complete_response = ""
        async for part in response:  # Iterate over the async generator
            if part is not None:
                if isinstance(part, Delta):
                    if ("tool_calls" in part) and (part['tool_calls'] is not None):
                        async for code_execution_message in CONVO.execute_tool_calls(part["tool_calls"]):
                            if code_execution_message:  # the penultimate response contains the code, stdout, stderr
                                await streaming_response.stream_token(code_execution_message)

                    elif ("content" in part) and (part['content'] is not None):
                        complete_response += part['content']
                        await streaming_response.stream_token(part["content"])

                    if complete_response != "":
                        await CONVO.add_assistant_msg(msg=complete_response)

        # After the entire response is streamed
        # streaming_response.content = complete_response
        # await CONVO.add_assistant_msg(msg=complete_response)
        await streaming_response.update()

    # Cache updated conversation
    cl.user_session.set("CONVO", CONVO)

    # Update and display token count (optional)
    TOTAL_TOKENS = CONVO.total_tokens
    token_count_message = cl.Message(content=f"Total tokens used: {TOTAL_TOKENS}")
    await token_count_message.send()


@cl.on_chat_start
async def on_chat_start():
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
        CONVO = init_convo(
            context_code="\n\n".join(all_code),
            user_question=first_msg["output"],
            model_name=ModelNames.GPT_3_5_TURBO.value,
        )

        # Cache the updated conversation, before you probe_llm, cause async magic
        cl.user_session.set("CONVO", CONVO)

        # Didn't know what to call this step, "probing/inferring" the LLM
        # with a chat prompt seems the most logical.
        await probe_llm()


@cl.on_message
async def on_message(message: cl.Message):
    # Get cached object
    CONVO: Conversation = cl.user_session.get("CONVO")

    # Ingest user message
    CONVO.add_user_msg(msg=message.content)

    # Cache the updated conversation, before you probe_llm, cause async magic
    cl.user_session.set("CONVO", CONVO)

    # Probe llm
    await probe_llm(from_user=True)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
