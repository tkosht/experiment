from inspect import signature

import gradio as gr
import typer
from omegaconf import DictConfig

from app.langchain.component.agent_bot import SimpleBot


def _init(history: list[tuple[str, str]], text: str):
    history = history + [(text, None)]
    return history, "", ""


def _main(params: DictConfig):
    _prompt_example = (
        "Download the langchain.com webpage and grep for all urls. "
        "Return only a sorted list of them. Be sure to use double quotes."
    )
    #     _prompt_default = """titanic dataset をダウンロードして(data/titanic.csv として保存し)、
    # scikit-learn の LightGBM を使ってクラス分類する python コードを作成して実行し、精度指標値を出力し確認する。
    # そして、検証用データの精度指標値を最高精度になるように改善する。 精度が下がったら、修正前のコードを基準に改善をすること。
    #
    # 尚、最高精度を目指すため、特徴量エンジニアリングなどは、https://qiita.com/jun40vn/items/d8a1f71fae680589e05c を参考にする。
    # 最後に、‘result/titanic.py’ というローカルファイルに上書き保存し、エラーがないことを実際に実行して確認する。
    #
    # これらは、python_repl ツール または bash/terminal ツール のいずれかのツールのみを使って試行し実現してください。
    # """
    _prompt_default = """terminal ツールを使ってtitanic dataset をダウンロードして(data/titanic.csv として保存し)、
scikit-learn の LightGBM を使ってクラス分類し精度指標値を出力する Python コードを作成し python_repl ツールを使って実際に実行結果をシミュレーションし確認します。
その後、精度を上げるためのパラメータチューニングや特徴量エンジニアリングなどで精度指標値が83%以上になるようにWeb上の叡智を活用して改善(python_repl ツールでシミュレーション)を繰り返します。
例えば "KaggleチュートリアルTitanicで上位2%以内に入るノウハウ" などのWebサイトが役に立ちます。参考にしましょう。
精度が不十分だったりエラーが発生したら修正・改善(python_repl ツールを使って実際の実行結果のシミュレーション)を繰り返します。
エラーを完全に解消し精度が十分に達成できたら、その Python コードを ‘result/titanic.py’ というローカルファイルに上書き保存してください。
"""  # noqa

    # 精度を上げるためのパラメータチューニングと特徴量エンジニアリングなどで制度指標値を改善します。
    # 本依頼の実行開始時と終了時の時刻を忘れずに具体的に教えてください。
    # あなたが、Action/$JSON_BLOB フォーマットを忘れずに使うことで、利用可能なツールを実行できることを絶対に忘れないでください。

    bot = SimpleBot()

    with gr.Blocks() as demo:
        with gr.Tab("Conversation"):
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(
                        [], label="assistant", elem_id="demobot", height=405
                    )
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder=_prompt_example,
                        value=_prompt_default,
                        container=False,
                    )

                with gr.Column():
                    log_area = gr.TextArea(
                        lines=21,
                        max_lines=21,
                        show_label=False,
                        label="log",
                        placeholder="",
                        value="",
                        container=False,
                    )
                    with gr.Row():
                        btn = gr.Button(value="update agent log")
                        btn.click(
                            bot.gr_update_text, inputs=[log_area], outputs=[log_area]
                        )

        with gr.Tab("Context"):
            with gr.Row():
                with gr.Column():
                    ctx_area = gr.TextArea(
                        lines=21,
                        max_lines=21,
                        show_label=False,
                        label="context",
                        placeholder="",
                        value="",
                        container=False,
                    )
                    with gr.Row():
                        btn = gr.Button(value="clear context")
                        btn.click(
                            bot.gr_clear_context, inputs=[ctx_area], outputs=[ctx_area]
                        )

        with gr.Tab("Setting"):
            with gr.Row():
                model_dd = gr.Dropdown(
                    [
                        "gpt-3.5-turbo-16k",
                        "gpt-3.5-turbo",
                        "gpt-4-0314",
                        "gpt-4",
                        "gpt-4-32k-0314",
                        "gpt-4-32k",
                    ],
                    value="gpt-3.5-turbo-16k",
                    label="chat model",
                    info="you can choose the chat model.",
                )
                temperature_sl = gr.Slider(0, 100, 10, step=1, label="temperature (%)")
                max_iterations_sl = gr.Slider(0, 50, 30, step=1, label="max_iterations")

        txt.submit(_init, [chatbot, txt], [chatbot, txt]).then(
            bot.gr_clear_text, [], [log_area]
        ).then(
            bot.gr_chat,
            [chatbot, model_dd, temperature_sl, max_iterations_sl, ctx_area],
            [chatbot, ctx_area],
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
