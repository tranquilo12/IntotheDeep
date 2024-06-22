"""
Just a small file to generate all types of prompts
"""

import subprocess
from pathlib import Path

from jinja2 import BaseLoader, Environment

BASE_PATH = Path("/Users/shriramsunder/Projects/IntotheDeep")
LIB_PATH = Path("")
FILENAMES = [
    BASE_PATH / ele for ele in ["app.py", "models.py", "oai_types.py", "utils.py"]
]
# FILENAMES += [
#     Path(
#         "/Users/shriramsunder/Projects/IntotheDeep/.venv/lib/python3.12/site-packages/chainlit/step.py"
#     )
# ]


def get_starting_prompt(q: str) -> str:
    """
    A Function that returns a string for the all the files + task

    Parameters
    -------
    q: str, The main question for this prompt
    Returns
    -------
    str: All the files and the string for the task
    """
    # Create a jinja2 env with a string loader
    env = Environment(loader=BaseLoader())

    # Define the template
    template = """
You are a prominent python developer, with more than 100 years experience coding in python, and the chainlit library in python. 
Use the code below to understand the problem and answer it to the best of your knowledge.
Here is the reference documentation: ```https://docs.chainlit.io/concepts/step```

{% for filename in filenames %}
```python
# filename: {{ filename }}
{{ files[filename] }}
```
{% endfor %}

I've provided you with all the code for my project that talks to an LLM and streams the 
response back to a Chainlit frontend. All the code here is for showing off the streaming capabilities
of this frontend. I'm also sticking with LiteLLM's API as it helps me integrate all types of LLM responses 
neatly with OpenAI's API. 

{{ q }}
"""
    files = {}
    for filename in FILENAMES:
        filepath = BASE_PATH / filename

        with open(filepath, "r") as f:
            files[filename] = f.read()

    prompt = env.from_string(template).render(filenames=FILENAMES, files=files, q=q)
    return prompt


if __name__ == "__main__":
    q = """
I'm facing an issue where I'm unable to change the model name of a conversation before it's initialized. 
i.e when I start the app, I have to first go through the "on_chat_start" function, which doesn't have the capability 
to modify/accept the modification of the model name before the conversation class is initialized. Help me fix this.
"""
    result = get_starting_prompt(q=q)

    process = subprocess.Popen(
        "pbcopy", env={"LANG": "en_US.UTF-8"}, stdin=subprocess.PIPE
    )
    process.communicate(result.encode("UTF-8"))
    print("copied to clipboard!")
