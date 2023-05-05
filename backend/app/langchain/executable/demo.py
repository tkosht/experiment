from inspect import signature

import gradio as gr
import typer
from langchain.schema import AgentAction
from omegaconf import DictConfig

from app.langchain.component.agent_builder import build_agent
from app.langchain.component.agent_executor import CustomAgentExecutor


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def _bot(history: list[tuple],
         agent_executor: CustomAgentExecutor,
         intermediate_steps: list[tuple[AgentAction, str]]):

    query = history[-1][0]

    query_org = query
    max_retries = 5
    for _ in range(max_retries):
        try:
            answer = agent_executor.run(input=query, intermediate_steps=intermediate_steps)
            break
        except Exception as e:
            query = f"Observation: \nERROR: {str(e)}\n\nwith fix ERROR, \n\nHUMAN: {query_org}\nThougt:"
            answer = f"Error Occured: {e} / couldn't be fixed."
            if "This model's maximum context length is" in str(e):
                agent_executor.intermediate_steps = agent_executor.intermediate_steps[-1:]
        finally:
            intermediate_steps = agent_executor.intermediate_steps

    assert answer
    history[-1][1] = answer       # may be changed url to href
    return history


def _main(params: DictConfig):
    agent_executor = build_agent()
    intermediate_steps: list[tuple[AgentAction, str]] = []

    def bot(history):
        return _bot(history, agent_executor, intermediate_steps)

    def clear_context():
        intermediate_steps.clear()
        return

    with gr.Blocks() as demo:
        with gr.Tab("Conversation"):
            chatbot = gr.Chatbot([], label="assis_bot", elem_id="demobot").style(
                height=400
            )

            prompt_example = ("Download the langchain.com webpage and grep for all urls. "
                              "Return only a sorted list of them. Be sure to use double quotes.")
            # prompt_default = ("最近話題のニュースを教えて？")
            # prompt_default = ("徳川家康とは？")
            prompt_default = prompt_example
            with gr.Row():
                with gr.Column(scale=2):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder=prompt_example,
                        value=prompt_default,
                    ).style(container=False)
                btn = gr.Button(value="clear context")
                btn.click(clear_context, inputs=[], outputs=[])

        txt.submit(
            add_text, [chatbot, txt], [chatbot, txt]
        ).then(bot, chatbot, chatbot)

    if params.do_share:
        demo.launch(share=True, auth=("user", "user123"), server_name="0.0.0.0", server_port=7860)
    else:
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


# @from_config(params_file="conf/app.yml", root_key="/train")
# def config(cfg: DictConfig):
def config():
    cfg = DictConfig(dict(is_experiment=True, do_share=False))
    return cfg


def main(
    do_share: bool = None,
):
    s = signature(main)
    kwargs = {}
    for k in list(s.parameters):
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    params = config()  # use as default
    params.update(kwargs)
    return _main(params)


if __name__ == "__main__":
    typer.run(main)
