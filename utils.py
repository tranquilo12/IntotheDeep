"""
For everything that's not types or actual execution.
"""

import json
import os
import sys
from pathlib import PosixPath, WindowsPath
from typing import (Any, List, Optional, Union)

from dotenv import load_dotenv
from git import GitCommandError, Repo
from pydantic import BaseModel

from oai_types import Assistant, Conversation, System, User

load_dotenv()


#############################################
## For all generic utils functions
#############################################

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
		File paths, input from gradio component

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
	return [input_list[i: i + 2] for i in range(0, len(input_list), 2)]


#############################################
## For all the conversation related functions
## Includes the OpenAI and Anthropic API calls
#############################################


def init_convo(
	model_name: str,
	context_code: Union[str, Any],
	user_question: str,
) -> Conversation:
	"""
	Get the starting conversation for the assistant

	Parameters
	----------
	model_name: str
		The name of the model if you don't want the default value = gpt-3.5-turbo
	context_code : str
		Code provided by the user
	user_question : str
		Question provided by the user

	Returns
	-------
	Conversation
		Starting conversation for the assistant
	"""
	
	# Append the system message with a list of rules that are common to both
	system_message_base = "".join(
		[
			"You are #1 on the Stack Overflow community leaderboard. ",
			"Do not tell me that you're not capable of solving the problem. ",
			"You will figure a way out to solve the problem. ",
			"If you're asked to generate code, do so within the '```python' '```' markdown tags, as they'll be "
			"extracted into a JSON structure.",
		]
	)
	
	user_message_base = "\n\n".join(
		[
			f"""Here is the code I have so far, in between the "```python" and "```" tags:""",
			f"""```python\n{context_code if context_code is not None else "There is no code Provided"}\n```""",
			"""And here is my question about the code below: """,
			f"""```text\n{user_question}\n```""",
		],
	)
	
	# Create the conversation object
	return Conversation(
		model_name=model_name,
		messages_=[
			System("".join(system_message_base)),
			User(user_message_base),
		],
	)


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