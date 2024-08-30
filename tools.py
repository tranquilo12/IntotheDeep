from typing import AsyncGenerator

from oai_types import CodeExecutionContent, Conversation, Interpreter, TextContent
from utils import get_git_commit, get_latest_changes

convo = Conversation()


#############################################
######### Function calling related ##########
#############################################
async def generate_code(
    instructions: str, output=True
) -> AsyncGenerator[CodeExecutionContent, None]:
    """
    Generates Python code based on the provided instructions and yields CodeExecutionContent chunks.

    Parameters
    ----------
    instructions : str
        Instructions for generating the Python code.
    output : bool, optional
        Whether to yield the output. Defaults to True.

    Yields
    ------
    CodeExecutionContent
        The generated code wrapped in CodeExecutionContent.
    """
    messages = [
        {
            "role": "system",
            "content": f"You are a python code generator, generate only the required code according to the instructions provided, your response MUST be a single complete python code between the ```python ``` tags. Here are the instructions: \n{instructions}\n. Only respond with code",
        },
    ]
    response = convo.call_llm_no_context(messages)

    accumulated_message = ""
    async for partial_message in response:
        accumulated_message = partial_message
        # Extract code from the response
        code = accumulated_message.lstrip("```python").rstrip("```")
        if output:
            yield CodeExecutionContent(code=code, stdout="")
        yield code


async def execute_code_locally(
    code: str, interpreter: Interpreter
) -> AsyncGenerator[CodeExecutionContent, None]:
    """
    Executes Python code using the Interpreter and yields CodeExecutionContent chunks.

    Parameters
    ----------
    code : str
        The Python code to execute.
    interpreter : Interpreter
        The interpreter to run the code.

    Yields
    ------
    CodeExecutionContent
        The execution output wrapped in CodeExecutionContent.
    """
    code = code.lstrip("```python").rstrip("```")
    stdout, stderr = await interpreter.run(code)

    # Split the stdout into lines and yield each line as a separate chunk
    for line in stdout.splitlines():
        yield CodeExecutionContent(code=code, stdout=line, stderr="")

    # If there's any stderr, yield it as a final chunk
    if stderr:
        yield CodeExecutionContent(code=code, stdout="", stderr=stderr)


async def debug_code(
    code: str, interpreter: Interpreter, output=True
) -> AsyncGenerator[CodeExecutionContent, None]:
    """
    Debugs Python code using the Interpreter and yields CodeExecutionContent chunks.

    Parameters
    ----------
    code : str
        The Python code to debug.
    interpreter : Interpreter
        The interpreter to run the code.
    output : bool, optional
        Whether to yield the output. Defaults to True.

    Yields
    ------
    CodeExecutionContent
        The debugged code and its execution output wrapped in CodeExecutionContent.
    """
    code = code.lstrip("```python").rstrip("```")
    max_attempts = 3
    attempt = 0
    while attempt < max_attempts:
        print(f"attempt: {attempt}")
        stdout, stderr = await interpreter.run(code)
        attempt += 1
        if stderr:
            instructions = (
                f"Code to fix: ```python\n{code}\n``` Error: ```python\n{stderr}\n```"
            )
            async for new_code in generate_code(instructions, output=False):
                code = new_code
        else:
            if output:
                for line in stdout.splitlines():
                    yield CodeExecutionContent(code=code, stdout=line, stderr="")
            else:
                yield code, stdout
            break


async def data_analyst(
    file_path: str, instructions: str = None
) -> AsyncGenerator[TextContent, None]:
    """
    Performs data analysis on the data file provided.

    Parameters
    ----------
    file_path : str
        Path to the data file.
    instructions : str, optional
        Specific instructions for the data analysis. Defaults to None.

    Yields
    ------
    TextContent
        The analysis summary wrapped in TextContent.
    """
    instructions = f"You are an expert data analyst specializing in the semiconductor industry. Using the data provided in the file at \n{file_path}\n, generate a complete Python script for data analysis using only pandas and numpy, avoid visualizations. Follow these specific instructions: \n{instructions}\n. If no instructions are provided, use your expertise to perform comprehensive analysis, and any relevant statistical analysis. Print all the outputs."
    interpreter = Interpreter()
    async for code in generate_code(instructions, output=False):
        gen_code = code
    async for new_code, out in debug_code(gen_code, interpreter, output=False):
        corrected_code, stdout = new_code, out
    messages = [
        {
            "role": "system",
            "content": f"Please review the data analysis output shown below: ```python\n{stdout}\n``` Provide a comprehensive and exhaustive summary of the key statistics and data insights. Include details that would be helpful for replicating this data such as structure, distribution, relation among different features, and any other noteworthy findings.",
        },
    ]
    response = convo.call_llm_no_context(messages)
    analysis = ""
    async for partial_message in response:
        analysis = partial_message
        yield TextContent(text=analysis)


async def data_generator(instructions) -> AsyncGenerator[CodeExecutionContent, None]:
    """
    Generates synthetic data based on the provided analysis instructions.

    Parameters
    ----------
    instructions : str
        The analysis summary to base the synthetic data generation on.

    Yields
    ------
    CodeExecutionContent
        The generated code for synthetic data wrapped in CodeExecutionContent.
    """
    messages = [
        {
            "role": "system",
            "content": f"""
                You are given a summary analysis of a dataset that includes details such as column types, distributions, statistical properties (e.g., mean, variance), and correlations between variables.\n
                Your task is to generate a Python code that replicates the structure and statistical distribution of this data as accurately as possible according to this analysis:```text\n{instructions}\n```\n
                The synthetic data should have 10000 data points (rows) and should reflect the column relationships, and distribution characteristics as described in the analysis. 
                Save the generated synthetic data as a CSV file.\n
                """,
        },
    ]
    async for code_content in generate_code(messages, output=True):
        yield code_content


async def get_latest_changes_within_git(
    root_path: str,
) -> AsyncGenerator[TextContent, None]:
    """
    Get the latest changes within a git repository.

    Parameters
    ----------
    root_path : str
        Path to the root of the git repository.

    Yields
    ------
    TextContent
        The summary of the latest changes wrapped in TextContent.
    """
    diffs, diffsummary = get_latest_changes(root_path)
    yield TextContent(text=diffsummary)


async def get_git_commit_prompt(root_path: str) -> AsyncGenerator[TextContent, None]:
    """
    Get the git commit prompt.

    Parameters
    ----------
    root_path : str
        Path to the root of the git repository.

    Yields
    ------
    TextContent
        The generated git commit message wrapped in TextContent.
    """
    git_diff, diffsummary = get_latest_changes(root_path)
    conversation = get_git_commit(git_diff)
    messages = conversation.to_dict()
    response = convo.call_llm_no_context(messages)

    commit_message = ""
    async for partial_message in response:
        commit_message = partial_message
        yield TextContent(text=commit_message)
