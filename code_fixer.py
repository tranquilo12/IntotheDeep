import os
from openai import AzureOpenAI
import gradio as gr

from interpreter import generate_and_debug_python_code

client = AzureOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
    api_key=os.getenv("AZURE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2024-02-15-preview",
)


def chatty(input, history):
    code, output, error = generate_and_debug_python_code(
        input, client, max_tokens=1000, model_name="gpt-4-0125-preview", max_attempts=5
    )
    formatted_code = "```python\n" + code + "\n```"
    error = "```python\n" + error + "\n```"
    formatted_output = "```python\n" + output + "\n```"
    if error:
        response = f"{formatted_code}\nOutput:\n{formatted_output}"
    else:
        response = f"{formatted_code}\nExecution Unsuccessful:\n{error}"
    return response


block = gr.Blocks()
with block:
    with gr.Row():
        with gr.Column():
            chat_interface = gr.ChatInterface(
                fn=chatty,
                fill_height=True,
                chatbot=gr.Chatbot(height=1000, render=False),
            )

block.launch()
