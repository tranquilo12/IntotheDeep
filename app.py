from typing import Any, List, Optional, Union

import chainlit as cl
from chainlit.input_widget import Select, Switch
from chainlit.types import AskFileResponse
from dotenv import load_dotenv

from utils.eventhandler import ChainlitEventHandler
from src.types import Conversation, Dict, System, User
from src.models import ModelNames


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """
    Authenticate the user based on username and password.

    Parameters
    ----------
    username : str
        The username of the user.
    password : str
        The password of the user.

    Returns
    -------
    cl.User or None
        Returns a `cl.User` object if authentication is successful, otherwise None.
    """
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


# Global variable to store the file actions message
file_actions_message = None
file_mapping: Dict[str, str] = {}  # Maps anonymized filenames to actual filenames

load_dotenv()


def init_convo(
    model_name: str,
    context_code: Union[str, Any],
    user_question: str,
) -> Conversation:
    """
    Initialize a conversation with the given parameters.

    Parameters
    ----------
    model_name : str
        The name of the model to use for the conversation.
    context_code : Union[str, Any]
        The context code to provide to the model.
    user_question : str
        The user's question to start the conversation.

    Returns
    -------
    Conversation
        The initialized conversation object.
    """
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
        model_name=model_name,
        messages_=[
            System(system_message_base),
            User(user_message_base),
        ],
        hist_path="chat_log.json",
        code_path="code_blocks.json",
    )


#############################################
########## Helper Functions (Convo) #########
#############################################
async def run_conversation(max_tokens: int = 4000):
    """
    Run the conversation with the specified maximum tokens.

    Parameters
    ----------
    max_tokens : int, optional
        The maximum number of tokens to use for the conversation (default is 4000).
    """
    CONVO: Conversation = cl.user_session.get("CONVO")

    event_handler: ChainlitEventHandler = ChainlitEventHandler(CONVO)
    await CONVO.call_llm(max_tokens=max_tokens, event_handler=event_handler)

    last_chat_message = CONVO.messages[-1]
    # Log the chat message
    CONVO.log_chat_message(last_chat_message.content.text)

    # Extract and append code from the chat message
    CONVO.extract_and_append_code(last_chat_message.content.text)

    # Ensure the conversation is updated in the user session
    cl.user_session.set("CONVO", CONVO)


async def generate_settings(
    preset_model: str, actual_filenames: List[str]
) -> Dict[str, str]:
    """
    Generate chat settings based on the preset model and actual filenames.

    Parameters
    ----------
    preset_model : str
        The preset model to use for the settings.
    actual_filenames : List[str]
        The list of actual filenames.

    Returns
    -------
    Dict[str, str]
        The generated settings as a dictionary.
    """
    return await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
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


@cl.on_chat_start
async def on_chat_start():
    """
    Handle the chat start event by initializing the conversation and settings.
    """
    global file_mapping

    preset: Dict[str, str] = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=ModelNames.all_to_list(),
                initial_value=ModelNames.AZURE_GPT_35_TURBO.value,
            ),
        ]
    ).send()

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

    all_code = []
    file_paths = []
    file_mapping = {}  # Reset file_mapping
    for py_f in files:
        with open(py_f.path, "r", encoding="utf-8") as f:
            code = f.read()
            formatted_code = f"### filename: {py_f.name} ###\n\n{code}\n\n###"
            all_code.append(formatted_code)
            file_paths.append(py_f.path)
            file_mapping[py_f.path] = (
                py_f.name
            )  # Map anonymized path to actual filename

    first_msg: Optional[cl.Message] = await cl.AskUserMessage(
        content="What do you want to do with these uploaded files?",
        timeout=60,
    ).send()

    actual_filenames = []
    for actual_filename in file_mapping.values():
        actual_filenames.append(actual_filename)
    preset_model = preset["Model"]
    settings: Dict[str, str] = await generate_settings(preset_model, actual_filenames)
    if first_msg:
        CONVO: Conversation = init_convo(
            model_name=settings["Model"],
            context_code="".join(all_code),
            user_question=first_msg["output"],  # type: ignore
        )
        CONVO.files = file_paths
        cl.user_session.set("CONVO", CONVO)
        CONVO.init_log()
        first_user_message = first_msg["output"]
        CONVO.log_user_message(first_user_message)

        await run_conversation()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages and update the conversation.

    Parameters
    ----------
    message : cl.Message
        The incoming message to handle.
    """

    CONVO: Conversation = cl.user_session.get("CONVO")
    # Check if there are any files attached to the message
    if message.elements:
        # Filter for .py files
        py_files = [file for file in message.elements if file.name.endswith(".py")]

        # If there are .py files, read their content
        if py_files:
            for py_file in py_files:
                with open(py_file.path, "r", encoding="utf-8") as f:
                    CONVO.add_user_msg(
                        f"Contents of `{py_file.name}`:\n\n```python\n{f.read()}\n```"
                    )

    CONVO.add_user_msg(message.content)
    CONVO.log_chat_message(message.content)
    cl.user_session.set("CONVO", CONVO)
    await run_conversation()


async def reset_settings():
    """
    Reset the chat settings based on the current conversation.
    """
    actual_filenames = []
    for actual_filename in file_mapping.values():
        actual_filenames.append(actual_filename)
    CONVO: Conversation = cl.user_session.get("CONVO")
    preset_model = CONVO.model_name
    await generate_settings(preset_model, actual_filenames)


@cl.on_settings_update
async def on_settings_update(settings):
    """
    Handle settings update events and update the conversation settings.

    Parameters
    ----------
    settings : dict
        The updated settings.
    """
    # Get cached object
    CONVO: Conversation = cl.user_session.get("CONVO")

    if CONVO is not None:
        # Update the conversation's model name
        CONVO.model_name = settings["Model"]
        # Cache the updated conversation, before you probe_llm, cause async magic
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
            for anonymized, actual in file_mapping.items():
                if actual == actual_filename:
                    anonymized_filename = anonymized

            CONVO: Conversation = cl.user_session.get("CONVO")
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
