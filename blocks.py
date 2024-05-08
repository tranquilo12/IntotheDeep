import os
import subprocess
import gradio as gr
from typing import Optional


def auto_correct_path(input_path):
    if input_path is None:
        return None

    system = os.name
    if system == "nt":  # Windows
        corrected_path = input_path.replace("/", "\\\\")
        corrected_path = corrected_path.replace("\\", "\\\\")
    else:  # Unix, Linux, MacOS
        corrected_path = input_path.replace("\\", "/")
    return corrected_path


def get_chosen_files(repo_path: Optional[str]):
    return gr.FileExplorer(
        label="Repository Path",
        glob="**/*.py",
        file_count="multiple",
        root_dir=auto_correct_path(repo_path),
        show_label=False,
    )


def select_chat_window():
    return gr.Tabs(selected=1)


def get_repo_path_textbox(placeholder=None, scale=None):
    default_placeholder = "Provide Repository Path"

    return gr.Textbox(
        scale=6 if scale is None else scale,
        label="Repo Path",
        type="text",
        interactive=True,
        placeholder=default_placeholder if placeholder is None else placeholder,
        render=False,
    )


def get_env_dropbox(env_path: str, placeholder=None):
    default_placeholder = "Environments Found"

    # Clean up the path
    environments = []
    env_path = auto_correct_path(env_path)

    if env_path is not None:
        if not os.path.exists(env_path):
            raise FileNotFoundError(f"Directory '{env_path}' not found.")

        # Find conda environments
        conda_env_dirs = [
            os.path.join(os.path.expanduser("~"), "miniconda3", "envs"),
            os.path.join(os.path.expanduser("~"), "anaconda3", "envs"),
        ]
        for conda_env_dir in conda_env_dirs:
            if os.path.exists(conda_env_dir):
                environments.extend(
                    [
                        os.path.join(conda_env_dir, env)
                        for env in os.listdir(conda_env_dir)
                    ]
                )

        # Find virtual environments
        for root, dirs, _ in os.walk(env_path):
            for dir_name in dirs:
                if dir_name == ".venv" or dir_name == "venv":
                    environments.append(os.path.join(root, dir_name))

    return gr.Dropdown(
        choices=environments,
        label=default_placeholder if placeholder is None else placeholder,
        render=False,
        multiselect=False,
    )


def get_env_path_textbox(placeholder=None, scale=None):
    default_placeholder = "Path to find all the environments"

    return gr.Textbox(
        scale=6 if scale is None else scale,
        label="Env Base Path",
        type="text",
        interactive=True,
        placeholder=default_placeholder if placeholder is None else placeholder,
        render=False,
    )


def get_max_tokens_slider(scale=None):
    return gr.Slider(
        scale=2 if scale is None else scale,
        minimum=100,
        maximum=5000,
        value=1200,
        step=100,
        interactive=True,
        label="Max Tokens",
        info="The maximum number of tokens to generate.",
        render=False,
    )


def get_model_names_dropdown():
    return gr.Dropdown(
        choices=["gpt-4-0125-preview", "gpt-3.5-turbo"],
        value="gpt-3.5-turbo",
        label="Model Name",
        render=False,
        multiselect=False,
    )


def get_prompt_options():
    return gr.CheckboxGroup(
        scale=1,
        label="Prompt Options",
        choices=["Code Only", "Efficient", "Add Assistant Message", "Explain"],
        render=False,
    )


def get_new_system_prompt():
    return gr.Textbox(
        scale=1,
        label="New System Message",
        type="text",
        interactive=True,
        placeholder="Alternative system message",
        render=False,
    )


def stage_changes():
    try:
        subprocess.run(["git", "add", "."], check=True)
    except subprocess.CalledProcessError:
        raise Exception("Failed to stage changes.")
