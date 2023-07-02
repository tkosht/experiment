from inspect import signature

import gradio as gr
import typer
from omegaconf import DictConfig

from app.semantic_kernel.semantic_bot import SemanticBot


def _init(history: list[tuple[str, str]], text: str):
    history = history + [(text, None)]
    return history, "", ""


def _main(params: DictConfig):
    bot = SemanticBot()
    default_query = "LLMについて教えて？"

    with gr.Blocks() as demo:
        with gr.Tab("Conversation"):
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(
                        [], label="assistant", elem_id="demobot"
                    ).style(height=405)
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="入力してね〜",
                        value=default_query,
                    ).style(container=False)

        with gr.Tab("Setting"):
            with gr.Row():
                model_dd = gr.Dropdown(
                    [
                        "gpt-3.5-turbo",
                        "gpt-3.5-turbo-0613",
                        "gpt-3.5-turbo-16k",
                        "gpt-3.5-turbo-16k-0613",
                        "gpt-4",
                        "gpt-4-0613",
                        "gpt-4-32k",
                        "gpt-4-32k-0613",
                    ],
                    value="gpt-3.5-turbo",
                    label="chat model",
                    info="you can choose the chat model.",
                )
                temperature_sl = gr.Slider(0, 200, 1, step=1, label="temperature (%)")

        txt.submit(_init, [chatbot, txt], [chatbot, txt]).then(
            bot.gr_chat,
            [chatbot, model_dd, temperature_sl],
            [chatbot],
        )

    if params.do_share:
        demo.launch(
            share=True,
            auth=("user", "user123"),
            server_name="0.0.0.0",
            server_port=7860,
        )
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
