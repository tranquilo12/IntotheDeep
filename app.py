import chainlit as cl
from dotenv import load_dotenv
from utils import init_convo
from oai_types import Conversation
from models import ModelNames

load_dotenv()


# Global variable to track token count across sessions (optional)
TOTAL_TOKENS: int = 0


async def probe_llm(max_tokens: int = 500):
    global TOTAL_TOKENS  # Reference global token counter
    CONVO: Conversation = cl.user_session.get("CONVO")  # Get cached object

    # Send user message immediately
    user_message = cl.Message(content=CONVO.messages_[-1].content.text)
    await user_message.send()

    # Get response from LLM (streamed or not)
    response = CONVO.call_llm(max_tokens=max_tokens, stream=True)

    # Handle streamed or non-streamed responses
    if isinstance(response, str):  # Non-streamed response
        await CONVO.add_assistant_msg(msg=response)
        assistant_message = cl.Message(content=response)
        await assistant_message.send()  # Send assistant message
    else:  # Streamed response
        streaming_response = cl.Message(content="")
        await streaming_response.send()
        complete_response = ""
        async for part in response:
            if part is not None:
                if isinstance(part, dict) and "function_call" in part:
                    await CONVO.process_tool_calls(part)
                else:
                    await streaming_response.stream_token(part)
                    complete_response += part
            else:
                complete_response += "<$None$> "

        # After the entire response is streamed
        streaming_response.content = complete_response
        await CONVO.add_assistant_msg(msg=complete_response)
        await streaming_response.update()

    # Cache the updated conversation
    cl.user_session.set("CONVO", CONVO)

    # Update and display token count (optional)
    TOTAL_TOKENS = CONVO.total_tokens
    token_count_message = cl.Message(content=f"Total tokens used: {TOTAL_TOKENS}")
    await token_count_message.send()


@cl.on_chat_start
async def on_chat_start():
    files = None
    while files == None:
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

    first_msg: cl.Message = await cl.AskUserMessage(
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

        # Didn't know what to call this step, "probing" the LLM
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
    await probe_llm()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
