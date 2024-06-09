import os
import sys
import json

from litellm import acompletion
from pathlib import WindowsPath, PosixPath

from typing import (
    AsyncIterator,
    Optional,
    Union,
    List,
    Dict,
    Any,
)

from pydantic import BaseModel
from dotenv import load_dotenv
from git import Repo, GitCommandError

from oai_types import Conversation, User, Assistant, System

load_dotenv()

#############################################
## For all generic utils functions
#############################################

import more_itertools as mit


def chunk_words(text, chunk_size, overlap):
    """
    # Example usage
    text = (
        "This is an example of a text that needs to be chunked into overlapping segments."
    )
    chunk_size = 5
    overlap = 2

    chunks = chunk_words(text, chunk_size, overlap)
    chunks_with_text = [" ".join(chunk).strip() for chunk in chunks]
    for chunk in chunks_with_text:
        print(chunk)

    Parameters:
    ----------
    text: str = The text to be split
    chunk_size: int = Self explanatory
    overlap: int = Also, self explanatory

    Returns:
    ---------
    List = A list of chunks of the original string.
    """
    words = text.split()
    step = chunk_size - overlap
    return list(mit.windowed(words, n=chunk_size, step=step, fillvalue=""))


def str_to_path(path: str | List) -> Optional[WindowsPath | PosixPath]:
    """
    Convert a string to a path object, based on the OS
    Parameters
    ----------
    path: str | List

    Returns
    -------
    Optional[WindowsPath | PosixPath]
    """

    if isinstance(path, list):
        if sys.platform == "win32":
            path = WindowsPath("\\\\".join(path))
        else:
            path = PosixPath("/".join(path))
    else:
        if sys.platform == "win32":
            path = WindowsPath(path)
        else:
            path = PosixPath(path)

    return path


def read_all_files(file_paths: List) -> Optional[str]:
    """
    Read all files in a list of file paths and return a string with all the contents

    Parameters
    ----------
    file_paths : List
    List of file paths, input from gradio component

    Returns
    -------
    Optional[str]
        String with all the contents of the files
    """

    if file_paths:
        all_content = ""

        for filename in file_paths:
            # Check if the file is a python file, possibly add more file types
            if filename.endswith(".py"):
                with open(filename, "r") as f:
                    content = f.read()
                    all_content += (
                        f"\n# FILENAME : {filename}"
                        + "\n\n"
                        + content
                        + "\n"
                        + "#" * 10
                        + "\n"
                    )

        # Nothing else will be returned if it's empty
        return all_content


def convert_to_pairs(input_list: List) -> List[List[str]]:
    """
    Takes in a list of strings and converts it to a list of pairs of strings

    Parameters
    ----------
    input_list : List
       Of strings

    Returns
    -------
    List[List[str]]
        List of pairs of strings

    """
    return [input_list[i : i + 2] for i in range(0, len(input_list), 2)]


#############################################
## For all the conversation related functions
## Includes the OpenAI and Anthropic API calls
#############################################


def init_convo(
    code_only: bool,
    model_name: str,
    efficient: bool,
    explain: bool,
    context_code: Union[str, Any],
    user_question: str,
) -> Conversation:
    """
    Get the starting conversation for the assistant

    Parameters
    ----------
    code_only : bool
        Flag to indicate if the user wants to provide code only
    model_name: str
        The name of the model if you don't want the default value = gpt-3.5-turbo
    efficient: bool
        Flag to indicate whether the user wants the most efficient code from the assistant.
    explain: bool
        Flag to indicate whether the user wants an explanation from the assistant.
    context_code : str
        Code provided by the user
    user_question : str
        Question provided by the user

    Returns
    -------
    Conversation
        Starting conversation for the assistant
    """
    system_message_base = ["You are #1 on the Stack Overflow community leaderboard. "]

    if efficient:
        efficient_message = "If you provide code, it should be the most algorithmically efficient solution possible. "
        system_message_base.append(efficient_message)

    if explain:
        explain_message = "Tell the user that you're going to provide detailed explanations of the code. After every part explained, wait for the user's input before proceeding to the next part. "
        system_message_base.append(explain_message)

    if code_only:
        code_only_message = "You have been trained to only reply with the code. You will be provided with some context code, and possibly an error message. You will do your best to determine the root cause of the problem and return the properly formatted solution. Add type hints and docstring (numpy format). Always surround your replies with '```python' and '```' tags so its markdown formatted. Always wrap the entire code around the main() function and call the main function at the end of the code."
        system_message_base.append(code_only_message)

    # Append the system message with a list of rules that are common to both
    system_message_base += [
        "Do not tell me that you're not capable of solving the problem. ",
        "You will figure a way out to solve the problem. ",
    ]

    # Now get the standard user message
    if context_code is None:
        context_code = "There is no code provided"

    user_message_base = "\n\n".join(
        [
            f"Here is the code I have so far, in between the <code></code> tags: ",
            f"<code>{context_code}</code>",
            "And here is my question about the code, in between the <question></question> tags: ",
            f"<question>{user_question}</question>",
        ]
    )

    # Get all the messages of the conversation
    messages = [
        System("".join(system_message_base)),
        User(user_message_base),
    ]

    # Create the conversation object
    if model_name is not None:
        conversation = Conversation(messages_=messages, model_name=model_name)
    else:
        conversation = Conversation(messages_=messages)

    # Add the execute_code_locally function to the conversation
    # conversation.tools = [
    #     {
    #         "name": "execute_code_locally",
    #         "description": "Executes the provided Python code locally and returns the output and any error messages.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "code": {
    #                     "type": "string",
    #                     "description": "The Python code to execute.",
    #                 }
    #             },
    #             "required": ["code"],
    #         },
    #     }
    # ]

    return conversation


async def call_llm(
    conversation: Conversation,
    max_tokens: int,
    model_name: str,
    stream: bool = False,
) -> AsyncIterator | Dict[str, str]:
    """
    Call the Litellm API to get the response.

    Parameters
    ----------
    conversation: Conversation
        The raw input into the OpenAI API
    max_tokens : int
        Maximum number of tokens to generate
    model_name : str
        Name of the model to use
    stream: bool
        Boolean flag to indicate if the response should be streamed

    Returns
    -------
    Dict[str, str]
        Response from the OpenAI API
    """
    response = await acompletion(
        model=model_name,
        messages=conversation.to_dict(),
        stream=stream,
        max_tokens=max_tokens,
    )

    if stream:
        partial_message = ""
        async for chunk in response:
            if chunk.choices:
                if (chunk.choices[0].delta.content is not None) and (
                    len(chunk.choices[0].delta.content) > 0
                ):
                    partial_message += chunk.choices[0].delta.content
                    yield partial_message
    else:
        yield response.choices[0].message.content
        return


#############################################
## Helpful functions operating on the conv obj
## can move back and forth between convo and history
#############################################


def convert_convo_to_history(conversation: Conversation) -> List[List[str]]:
    """
    Convert the conversation to a list of strings

    Parameters
    ----------
    conversation : Conversation
        object

    Returns
    -------
    List[str]
        List of strings

    """
    messages: List[str] = [
        message.content[0].text for message in conversation.messages_
    ]
    pairs_of_messages: List[List[str]] = convert_to_pairs(messages)
    return pairs_of_messages


def convert_history_to_convo(
    history: List[List[str]],
    model_name: str = None,
) -> Conversation:
    """
    Convert the history to a conversation object

    Parameters
    ----------
    history : List[List[str]]
        List of pairs of strings

    model_name: str
        The name of the model if you don't want the default value = gpt-3.5-turbo
    Returns
    -------
    Conversation
    """
    # Depending on the discard_first_pair flag, the history will be appended with the starting messages
    if model_name is not None:
        conversation = Conversation(
            messages_=[System(history[0][0]), User(history[0][1])],
            model_name=model_name,
        )
    else:
        conversation = Conversation(
            messages_=[System(history[0][0]), User(history[0][1])]
        )

    # Add the history to the conversation
    for pair in history[1:]:
        conversation.messages_.append(Assistant(pair[0]))

        if len(pair) > 1:
            conversation.messages_.append(User(pair[1]))

    return conversation


#############################################
## For all the git related functions
#############################################


class GitFileDiff(BaseModel):
    filepath: Union[str, os.PathLike]
    diff: str


class AllGitFileDiffs(BaseModel):
    diffs: List[GitFileDiff]


def get_latest_changes_within_git(root_path: str | os.PathLike) -> List[GitFileDiff]:
    """
    Get the latest changes within a git repository

    Parameters
    ----------
    root_path : str
        Path to the git repository

    Returns
    -------
    AllGitFileDiffs
        Latest changes within the git repository
    """
    root_path = str_to_path(root_path)
    try:
        repo = Repo(root_path)
    except GitCommandError:
        raise ValueError("Invalid Git repository path")

    diffs = []
    staged_files = [item.a_path for item in repo.index.diff("HEAD")]

    for file in staged_files:
        try:
            diff = repo.git.diff("HEAD", file)
            diffs.append(GitFileDiff(filepath=file, diff=diff))
        except GitCommandError:
            pass

    return diffs


def get_git_commit_prompt(diff: GitFileDiff) -> Conversation:
    """
    Get the git commit prompt

    Parameters
    ----------
    diff : AllGitFileDiffs
        All the git file diffs

    Returns
    -------
    Conversation
        Git commit prompt
    """
    # Start the system message with a list of rules, it will be further
    # appended depending on the code_only flag
    system_message = System(
        "\n\n".join(
            [
                "Your only task is to provide a very comprehensive git commit message. ",
                "Try and be as detailed as possible, format it within points if needed. ",
                "You will be provided with an object of the structure: ",
                f"Git Diff Struct:",
                json.dumps(GitFileDiff.model_json_schema()),
            ]
        ),
    )

    # Get the formatted messages
    user_message = User(
        "\n\n".join(
            [
                f"Here is the git diff structure between the <gitDiff></gitDiff> tags: ",
                f"<gitDiff>{diff.model_dump_json()}</gitDiff>",
                "Give me a very comprehensive git commit message, in markdown. ",
                "Explain the benefits of the changes, and the drawbacks of the changes. ",
                "If they're just formatting changes, then say so, be succinct when needed. ",
            ]
        ),
    )

    # Create the assistant message
    assistant_message = Assistant("")
    # Create the conversation object
    return Conversation(messages_=[system_message, user_message, assistant_message])
