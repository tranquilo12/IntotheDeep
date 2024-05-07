import gradio as gr
from typing import List
from dotenv import load_dotenv

# The env variables are loaded from the .env file here,
# So all the imports after this will have access to the env variables
load_dotenv()
from blocks import (
    get_repo_path_textbox,
    get_max_tokens_slider,
    get_model_names_dropdown,
    get_chosen_files,
    get_new_system_prompt,
    get_prompt_options,
)

from utils import (
    gen_and_debug_python_code,
    read_all_files,
    init_convo,
    convert_history_to_convo,
    call_llm,
)


# Some css to make the app look better
css_code = """
/* Remove padding and margin for the Gradio app */
.gradio-container {
    padding: 0 !important;
    margin: auto !important;
    height: 100vh !important;
    max_width: 100vh !important;
    overflow: auto;      /* Allows scrolling inside the container if content exceeds the viewport */
}
/* Adjusting the images and button to utilize full space */
.cls {
    max-width: 100% !important;  /* Ensuring they don't overflow the container width */
    width: 100% !important;     /* Setting the width to the maximum available */
    margin: 0 !important;       /* Removing any default margins */
    padding: 0 !important;      /* Removing any default paddings */
}
"""


async def respond(
    msg: str,
    history: List,
    file_paths: List,
    model_name: str,
    max_tokens: int,
    new_system_prompt: str,
    prompt_options: List,
):
    # Convert the prompt options to bools
    prompt_options_dict = {
        "code_only": True if "Code Only" in prompt_options else False,
        "efficient": True if "Efficient" in prompt_options else False,
        "explain": True if "Explain" in prompt_options else False,
    }

    # If there is no history, initialize the conversation
    if len(history) == 0:
        convo = init_convo(
            user_question=msg,
            model_name=model_name,
            context_code=read_all_files(file_paths),
            **prompt_options_dict
        )
    else:  # Otherwise, we need to convert the history to a conversation
        convo = convert_history_to_convo(history=history, model_name=model_name)
        convo.add_user_msg(msg)

    # Update the system prompt, if needed
    if new_system_prompt is not None:
        convo.update_system_message(new_system_prompt)

    # If we've only requested for "code_only", then we're expecting only code
    # within the reply, so we extract and execute it in 'generate_and_debug_python_code'
    if prompt_options_dict["code_only"]:
        convo = await gen_and_debug_python_code(
            convo=convo,
            max_tokens=max_tokens,
        )
        yield convo.messages[-1].content.text

    else:  # Else, just call the LLM
        response = call_llm(
            conversation=convo,
            model_name=model_name,
            max_tokens=max_tokens,
            stream=True,
        )

        async for partial_message in response:
            yield partial_message


def toggle_chatbot_ui(state):
    state = not state
    return gr.update(visible=state), state


if __name__ == "__main__":
    with gr.Blocks(css=css_code) as demo:
        # Init the file explorer
        file_explorer = gr.FileExplorer(render=False, glob="**/*.py")

        # Define all the components here, render them later. Order matters for render.
        repo_path = get_repo_path_textbox()

        # For the max tokens
        max_tokens = get_max_tokens_slider()

        # For the model name
        model_names = get_model_names_dropdown()

        # Get the new system prompt modifier
        alt_system_message = get_new_system_prompt()

        # Get the prompt options
        prompt_options = get_prompt_options()

        # You want to ensure that it's rendering after the button is clicked.
        with gr.Tabs():
            with gr.TabItem("Chat with your Files", id=0):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, visible=True):
                        with gr.Row():
                            # Render row containing repository path
                            repo_path.render()

                            # Trigger the file explorer
                            choose_dir = gr.Button(
                                value="Submit Directory",
                                variant="primary",
                                size="sm",
                            )
                            choose_dir.click(
                                fn=get_chosen_files,
                                inputs=[repo_path],
                                outputs=[file_explorer],
                            ).then(
                                lambda: gr.update(variant="secondary"), [], [choose_dir]
                            )

                        with gr.Row():  # Render the file explorer window
                            file_explorer.render()

                        with gr.Row():  # Render the "choose files" button
                            choose_files = gr.Button(
                                value="Choose Files",
                                variant="primary",
                                size="sm",
                                scale=2,
                            )
                            choose_files.click(
                                fn=get_chosen_files,
                                inputs=[repo_path],
                                outputs=[file_explorer],
                            )

                    with gr.Column(
                        scale=4,
                        visible=False,
                    ) as chatbot:
                        with gr.Accordion(open=False, label="Model Parameters"):
                            # The accordian contains all the tunable params of the
                            # model. It has 2 rows, cause it's easier to see more of the
                            # system mesage like this.
                            with gr.Row(variant="panel"):
                                model_names.render()
                                max_tokens.render()
                                prompt_options.render()
                            with gr.Row(variant="panel"):
                                alt_system_message.render()

                        # Render the chat interface
                        gr.ChatInterface(
                            respond,  # Contains msg, history as default params included.
                            additional_inputs=[
                                file_explorer,
                                model_names,
                                max_tokens,
                                alt_system_message,
                                prompt_options,
                            ],
                            analytics_enabled=False,
                        ).queue()

                chatbot_state = gr.State(False)
                toggle_chatbot = gr.Button("Start Chatting", variant="primary")
                toggle_chatbot.click(
                    toggle_chatbot_ui, [chatbot_state], [chatbot, chatbot_state]
                ).then(lambda: gr.update(visible=False), [], [toggle_chatbot])

                # # with gr.Row(equal_height=True):
                #     # The entire stdout of the code ouput
                #     gr.TextArea()

            # with gr.TabItem("Git Commit Message", id=2):

            #     def git_commit_response(dir_path: os.PathLike | str, model_name: str):
            #         global stream_stop_flag
            #         stream_stop_flag = False
            #         diffs = get_latest_changes_within_git(dir_path)
            #         all_responses = ""

            #         for diff in diffs:
            #             convo = get_git_commit_prompt(diff)
            #             response = call_llm(
            #                 conversation=convo,
            #                 max_tokens=max_tokens,
            #                 model_name=model_name,
            #             )
            #             for all_responses in response:
            #                 if stream_stop_flag:
            #                     return
            #                 yield all_responses

            #     # Get the markdown component
            #     with gr.Row(variant="panel"):
            #         git_commit_message = gr.TextArea(
            #             label="Git Commit Message", visible=True
            #         )

            #     # Render the dropdown menu for selecting the model
            #     with gr.Row(variant="panel"):
            #         model_names = gr.Dropdown(
            #             label="Model Name",
            #             value="gpt-4-0125-preview",
            #             choices=ModelNames.oai_to_list(),
            #         )

            #     with gr.Row(variant="panel"):
            #         stage_all_changes = gr.Button(
            #             value="Stage All Changes",
            #             variant="primary",
            #             size="sm",
            #             scale=1,
            #         )
            #         stage_all_changes.click(
            #             fn=stage_changes,
            #         )

            #     # Render a button to generate the git commit message
            #     with gr.Row(variant="panel"):
            #         generate_commit_message = gr.Button(
            #             value="Generate Commit Message", variant="secondary", scale=1
            #         )

            #         generate_commit_message.click(
            #             fn=git_commit_response,
            #             inputs=[repo_path, model_names],
            #             outputs=[git_commit_message],
            #         )

            #         def stop_stream_handler():
            #             global stream_stop_flag
            #             stream_stop_flag = True

            #         stop_stream = gr.Button(
            #             value="Stop Stream", variant="stop", scale=1
            #         )
            #         stop_stream.click(fn=stop_stream_handler)

    demo.launch(height=1000)
