from inspect import signature

import gradio as gr
import typer
from omegaconf import DictConfig

from app.semantic_kernel.component.planner import CustomPlanner
from app.semantic_kernel.component.runner import SimpleRunner


def _init(
    history: list[tuple[str, str]], text: str
) -> tuple[list[tuple[str, str]], str]:
    history = history + [(text, None)]
    return history, ""


def _init_session(status: dict) -> dict:
    import uuid

    if "session_id" in status:
        return status

    status["session_id"] = str(uuid.uuid4())
    return status


async def _run(
    status: dict,
    history: list[tuple[str, str]],
    model_name: str,
    temperature_percent: int,  # in [0, 200]
    max_tokens: int = 1024,
) -> list[tuple[str, str]]:
    input_query: str = history[-1][0]
    temperature: float = temperature_percent / 100

    skill_dir = "./app/semantic_kernel/component/skills/"
    runner = SimpleRunner(
        planner=CustomPlanner(temperature=temperature, max_tokens=max_tokens),
        skill_dir=skill_dir,
        model_name=model_name,
    )
    context = "---\n\n".join([f"order: {q}\nanswer: {a}" for q, a in history[:-1]])
    context = context.replace("\n", "\n    ")
    user_query = f"""`これまでのユーザの依頼と回答(文脈)` を可能な範囲で踏まえて、`今回のユーザの依頼` に応えてください。

    # 今回のユーザの依頼
    <<<
    {input_query}
    >>>

    # これまでのユーザの依頼と回答(文脈)
    <<<
    {context if context else "なし"}
    >>>
"""

    response = await runner.do_run(user_query=user_query, n_retries=3)
    history[-1][1] = response
    return status, history


def _main(params: DictConfig):
    # default_query = "今日の川崎の天気を教えてくれませんか？"
    default_query = "今日の大規模言語モデル(LLM)に関するニュースを調べて情報源を含めて正確にわかりやすくまとめてくれますか？"

    with gr.Blocks() as demo:
        status = gr.State({})

        with gr.Tab("Conversation"):
            with gr.Row():
                chatbot = gr.Chatbot([], label="assistant", elem_id="demobot").style(
                    height=500
                )
            with gr.Row():
                with gr.Column():
                    txt = gr.TextArea(
                        show_label=False,
                        placeholder="入力してね〜",
                        value=default_query,
                        lines=5,
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

        txt.submit(_init_session, [status], [status]).then(
            _init, [chatbot, txt], [chatbot, txt]
        ).then(
            _run,
            [status, chatbot, model_dd, temperature_sl],
            [status, chatbot],
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
