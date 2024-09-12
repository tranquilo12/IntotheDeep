from typing import Any, List, Union

import chainlit as cl
from chainlit.input_widget import Select, Switch
from chainlit.types import AskFileResponse
from dotenv import load_dotenv

from src._types import Conversation, Dict, System, User
from src.models import ModelNames
from utils.eventhandler import ChainlitEventHandler

MAX_TOKENS: int = 4096


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    valid_users = {
        "anya": {"password": "jasthi", "role": "user"},
        "shriram": {"password": "sunder", "role": "user"},
    }

    if username in valid_users and password == valid_users[username]["password"]:
        return cl.User(
            identifier=username,
            metadata={"role": valid_users[username]["role"], "provider": "credentials"},
        )
    else:
        return None


file_actions_message = None
file_mapping: Dict[str, str] = {}

load_dotenv()


def init_convo(
    context_code: Union[str, Any],
    user_question: str,
    file_paths: list,
) -> Conversation:
    system_message_base = "".join(
        [
            "You are a helpful assistant equipped to handle multistep questions by using relevant functions. Follow a strict pattern:\n",
            "THOUGHT: Think step-by-step about which relevant function to call to progress towards the final answer.\n",
            "ACTION: Call a relevant function as the next step toward solving the problem, using arguments provided verbatim by the user or from the output of previous functions.\n"
            "OBSERVATION: Report the output of the function.\n",
            "Always prioritize calling a relevant function whenever applicable. If the query does not require function calling, respond appropriately to the user query",
            "The data is located within the /wta_training/data directory within the docker container that you're operating within.",
        ]
    )

    user_message_base = "\n\n".join(
        [
            f"""Here is the code I have so far, in between the "```python" and "```" tags:""",
            f"""```python\n{context_code if context_code is not None else "There is no code Provided"}\n```""",
            """And here is my question about the code below: """,
            f"""```text\n{user_question}\n```""",
        ]
    )

    return Conversation(
        messages_=[
            System(system_message_base),
            User(user_message_base),
        ],
        files=file_paths,
    )


async def run_conversation(max_tokens: int = 4000):
    CONVO: Conversation = cl.user_session.get("CONVO")
    event_handler: ChainlitEventHandler = cl.user_session.get("event_handler")
    await event_handler.call_llm(max_tokens=max_tokens)
    cl.user_session.set("CONVO", CONVO)


async def generate_settings(
    preset_model: str, actual_filenames: List[str]
) -> Dict[str, str]:
    return await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Models",
                values=ModelNames.all_to_list(),
                initial_value=preset_model,
            ),
            Select(
                id="File Removal",
                label="Remove Files",
                values=(
                    ["Please attach files"]
                    if not actual_filenames
                    else actual_filenames
                ),
            ),
            Switch(id="Download Chat", label="Download Chat History", initial=False),
        ]
    ).send()


def read_files_into_context(files: list):
    all_code, file_paths = [], []
    for py_f in files:
        with open(py_f.path, "r", encoding="utf-8") as f:
            all_code.append(f"### filename: {py_f.name} ###\n\n{f.read()}\n\n###")
            file_paths.append(py_f.path)
            file_mapping[py_f.path] = py_f.name
    return "\n\n".join(all_code), file_paths


@cl.on_chat_start
async def on_chat_start():
    # For removing files from context
    global file_mapping
    file_mapping = {}

    files: List[AskFileResponse] | None = await cl.AskFileMessage(
        content="Please upload python files only.",
        accept={
            "text/plain": [
                ".txt",
                ".py",
                ".env",
                ".html",
                ".css",
                ".js",
                ".csv",
                ".ipynb",
                ".json",
            ]
        },
        max_size_mb=10,
        timeout=240,
        max_files=10,
    ).send()

    if files:
        all_code, file_paths = read_files_into_context(files)

        first_msg: Dict = await cl.AskUserMessage(
            content="What do you want to do with these uploaded files?",
            timeout=360,
        ).send()
        if first_msg:
            CONVO: Conversation = init_convo(
                context_code=all_code,
                user_question=first_msg["output"],
                file_paths=file_paths,
            )
            event_handler: ChainlitEventHandler = ChainlitEventHandler(CONVO)
            await event_handler.call_llm(max_tokens=MAX_TOKENS)
            cl.user_session.set("event_handler", event_handler)


@cl.on_message
async def on_message(message: cl.Message):
    event_handler: ChainlitEventHandler = cl.user_session.get("event_handler")
    # Add message already sends the message to the UI, no need to send it again, it's handled in the
    # event handler stage.
    await event_handler.add_message(message=User(msg=message.content))
    await event_handler.call_llm(max_tokens=MAX_TOKENS)


async def reset_settings():
    actual_filenames = []
    for actual_filename in file_mapping.values():
        actual_filenames.append(actual_filename)
    CONVO: Conversation = cl.user_session.get("CONVO")
    preset_model = CONVO.model_name
    await generate_settings(preset_model, actual_filenames)


@cl.on_settings_update
async def on_settings_update(settings):
    CONVO: Conversation = cl.user_session.get("CONVO")

    if CONVO is not None:
        CONVO.model_name = settings["Model"]
        cl.user_session.set("CONVO", CONVO)

    if settings.get("Download Chat"):
        file = cl.File(
            name="chat_log.json", path="./chat_log.json", mime="application/json"
        )

        await cl.Message(
            content="Here's your conversation history. Click to download!",
            elements=[file],
        ).send()
        await reset_settings()

    if settings.get("File Removal"):
        selected_file = settings["File Removal"]
        if selected_file not in file_mapping.values():
            return
        else:
            actual_filename = selected_file
            anonymized_filename = None
            for anonymized, actual in file_mapping.items():
                if actual == actual_filename:
                    anonymized_filename = anonymized

            if anonymized_filename:
                CONVO.remove_file(anonymized_filename)
                del file_mapping[anonymized_filename]
                cl.user_session.set("CONVO", CONVO)
                await cl.Message(
                    f"File '{actual_filename}' removed from the conversation context."
                ).send()
                await reset_settings()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
