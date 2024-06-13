import os
import chainlit as cl
from dotenv import load_dotenv
from utils import init_convo
from oai_types import Conversation
from models import ModelNames

load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload python files only.",
            accept={"text/plain": [".txt", ".py"]},
            max_size_mb=10,
            timeout=240,
            max_files=4,
        ).send()

    msg = cl.Message(
        content=f"Processing {len(files)} file(s)...",
        disable_feedback=True,
    )
    await msg.send()

    all_code = []
    for py_f in files:
        with open(py_f.path, "r", encoding="utf-8") as f:
            code = f.read()
            formatted_code = f"### filename: {py_f.name} ###\n\n{code}\n\n###"
            all_code.append(formatted_code)

    CONVO = init_convo(
        context_code="\n\n".join(all_code),
        user_question=os.getenv("DEFAULT_CODE_Q"),
        model_name=ModelNames.GPT_3_5_TURBO.value,
        code_only=False,
        efficient=False,
        explain=False,
    )

    cl.user_session.set("CONVO", CONVO)


@cl.on_message
async def on_message(message: cl.Message):
    CONVO: Conversation = cl.user_session.get("CONVO")  # Get cached object
    CONVO.add_user_msg(msg=message.content)  # ingest user message

    # Instanciate empty message, for streaming into later
    streaming_response = cl.Message(content="")
    await streaming_response.send()

    # Start streaming!
    stream = CONVO.call_llm(stream=True, max_tokens=500)
    complete_response = []
    async for part in stream:
        complete_response += [part]
        await streaming_response.stream_token(part)

    # Update convo with assistant message, save it
    # await CONVO.add_assistant_msg(msg="".join(complete_response))
    streaming_response.content = CONVO.messages_[-1].content.text
    cl.user_session.set("CONVO", CONVO)

    # Need an element that keeps track of the total tokens used.
    elements = [cl.Text(name="Token Count", content=str(CONVO.total_tokens))]
    streaming_response.elements = elements

    await streaming_response.update()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
