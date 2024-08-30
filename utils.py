"""
For everything that's not types or actual execution.
"""

import json
import os
import sys
from pathlib import PosixPath, WindowsPath
from typing import List, Optional, Union, Tuple

from dotenv import load_dotenv
from git import GitCommandError, Repo
from pydantic import BaseModel

from oai_types import Conversation, System, User

load_dotenv()


#############################################
## For all generic utils functions
#############################################


def str_to_path(path: str | List) -> Optional[WindowsPath | PosixPath]:
    """
    Convert a string or list of strings to a path object, based on the OS.

    Parameters
    ----------
    path : Union[str, List[str]]
        The string or list of strings to convert to a path object.

    Returns
    -------
    Optional[Union[WindowsPath, PosixPath]]
        The corresponding path object based on the OS, or None if the input is invalid.
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


#############################################
## For all the git related functions
#############################################
class GitFileDiff(BaseModel):
    filepath: Union[str, os.PathLike]
    diff: str


class AllGitFileDiffs(BaseModel):
    diffs: List[GitFileDiff]


def get_latest_changes(root_path: str | os.PathLike) -> Tuple[List[GitFileDiff], str]:
    """
    Get the latest changes within a git repository.

    Parameters
    ----------
    root_path : Union[str, os.PathLike]
        Path to the git repository.

    Returns
    -------
    Tuple[List[GitFileDiff], str]
        A tuple containing the original diffs and their summaries.
    """
    root_path = str_to_path(root_path)
    try:
        repo = Repo(root_path)
    except GitCommandError:
        raise ValueError("Invalid Git repository path")

    diffs = []
    summaries = []
    staged_files = [item.a_path for item in repo.index.diff("HEAD")]

    for file in staged_files:
        try:
            diff = repo.git.diff("HEAD", file)
            summary = summarize_diff(diff)
            diffs.append(GitFileDiff(filepath=file, diff=diff))
            summaries.append(f"{file}: {summary}")
        except GitCommandError:
            pass

    summary_text = "\n".join(summaries)
    return diffs, summary_text


def summarize_diff(diff: str) -> str:
    """
    Generate a human-readable summary of the diff.

    Parameters
    ----------
    diff : str
        A string representing the diff output.

    Returns
    -------
    str
        A string containing a summary of the changes in the diff.
    """
    lines = diff.splitlines()
    added = sum(
        1 for line in lines if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in lines if line.startswith("-") and not line.startswith("---")
    )
    modified = len(lines) - added - removed

    description = []
    if added:
        description.append(f"{added} lines added")
    if removed:
        description.append(f"{removed} lines removed")
    if modified:
        description.append(f"{modified} lines modified")

    return ", ".join(description) if description else "No changes detected"


def get_git_commit(diff: GitFileDiff) -> Conversation:
    """
    Get the git commit prompt.

    Parameters
    ----------
    diff : GitFileDiff
        The git file diff.

    Returns
    -------
    Conversation
        Git commit prompt.
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
                f"<gitDiff>{diff}</gitDiff>",
                "Give me a very comprehensive git commit message, in markdown. ",
                "Explain the benefits of the changes, and the drawbacks of the changes. ",
                "If they're just formatting changes, then say so, be succinct when needed. ",
            ]
        ),
    )

    # Create the conversation object
    return Conversation(messages_=[system_message, user_message])
