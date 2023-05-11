import time
from inspect import signature

import gradio as gr
import typer
from langchain.schema import AgentAction
from omegaconf import DictConfig

from app.langchain.component.agent_builder import build_agent
from app.langchain.component.agent_executor import CustomAgentExecutor
from app.langchain.component.callbacks.simple import (BaseCallbackHandler,
                                                      TextCallbackHandler)


def _init(history: list[tuple[str, str]], text: str):
    history = history + [(text, None)]
    return history, "", ""


def _update_text(log_text: str, cb: TextCallbackHandler):
    return "\n".join(cb.log_texts)


def _bot(history: list[tuple],
         agent_executor: CustomAgentExecutor,
         intermediate_steps: list[tuple[AgentAction, str]],
         callbacks: list[BaseCallbackHandler] = []):

    query = history[-1][0]

    query_org = query
    max_retries = 10
    for _ in range(max_retries):
        try:
            answer = agent_executor.run(input=query, intermediate_steps=intermediate_steps, callbacks=callbacks)
            break
        except Exception as e:
            print("-" * 80)
            print(f"# {e.__repr__()} / in _bot()")
            print("-" * 80)
            query = f"Observation: \nERROR: {str(e)}\n\nwith fix this Error in other way, "
            f"\n\nHUMAN: {query_org}\nThougt:"
            if "This model's maximum context length is" in str(e):
                n_context = 1
                if len(agent_executor.intermediate_steps) <= n_context:
                    # NOTE: already 'n_context' intermediate_steps, but context length error. so, restart newly
                    query = query_org
                    agent_executor.intermediate_steps = []
                else:
                    agent_executor.intermediate_steps = agent_executor.intermediate_steps[-n_context:]
            else:
                # NOTE: retry newly
                query = query_org
                agent_executor.intermediate_steps = []
                answer = f"Error Occured: '{e}' / couldn't be fixed."

        finally:
            intermediate_steps = agent_executor.intermediate_steps
            print("<" * 80)
            print("Next:")
            print(f"{query=}")
            print("-" * 20)
            print(f"{len(agent_executor.intermediate_steps)=}")
            print(">" * 80)
        print("waiting retrying ... (a little sleeping)")
        time.sleep(1)

    assert answer
    history[-1][1] = answer       # may be changed url to href
    return history


def _main(params: DictConfig):
    _intermediate_steps: list[tuple[AgentAction, str]] = []

    _prompt_example = ("Download the langchain.com webpage and grep for all urls. "
                       "Return only a sorted list of them. Be sure to use double quotes.")
    # _prompt_default = _prompt_example
    # _prompt_default = ("徳川家康とは？")
    # _prompt_default = ("AIの最新ニュースを教えてちょ")
    _prompt_default = """titanic dataset をダウンロードして(data/titanic.csv として保存し)、
scikit-learn の LightGBM を使ってクラス分類する python コードを作成して実行し、精度指標値を出力し確認する。
そして、精度向上するために、https://qiita.com/jun40vn/items/d8a1f71fae680589e05c に記載されている特徴量エンジニアリング手法を参考にして精度改善し、
検証用データの精度指標値が90%以上まで改善すること。
その後、‘result/titanic.py’ というローカルファイルに上書き保存し、エラーがないことを確認する。

これらは、python_repl ツール または bash/terminal ツール のいずれかのツールのみを使って試行し実現してください。
"""
# 本依頼の実行開始時と終了時の時刻を忘れずに具体的に教えてください。
# あなたが、Action/$JSON_BLOB フォーマットを忘れずに使うことで、利用可能なツールを実行できることを絶対に忘れないでください。

    _callback = TextCallbackHandler(targets=["CustomAgentExecutor"])

    def bot(history: list[str], model_name: str, temperature_percent: int, max_iterations: int, context: str):
        agent_executor = build_agent(model_name, temperature=temperature_percent / 100, max_iterations=max_iterations)
        # h = _bot(history, agent_executor, _intermediate_steps, callbacks=[_callback])
        h = _bot(history, agent_executor, intermediate_steps=[], callbacks=[_callback])
        ctx = "\n".join([str(step[0]) for step in _intermediate_steps])
        return h, ctx

    def clear_context(context: str):
        _intermediate_steps.clear()
        return "\n".join(_intermediate_steps)

    def update_text(log_text: str):
        log = _update_text(log_text, _callback)
        return log

    def clear_text():
        _callback.log_texts.clear()
        log = _update_text("", _callback)
        return log

    with gr.Blocks() as demo:
        with gr.Tab("Conversation"):
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot([], label="assistant", elem_id="demobot").style(
                        height=405
                    )
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder=_prompt_example,
                        value=_prompt_default,
                    ).style(container=False)

                with gr.Column():
                    log_area = gr.TextArea(
                        lines=21,
                        max_lines=21,
                        show_label=False,
                        label="log",
                        placeholder="",
                        value="",
                    ).style(container=False)
                    with gr.Row():
                        btn = gr.Button(value="update agent log")
                        btn.click(update_text, inputs=[log_area], outputs=[log_area])

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
                    ).style(container=False)
                    with gr.Row():
                        btn = gr.Button(value="clear context")
                        btn.click(clear_context, inputs=[ctx_area], outputs=[ctx_area])

        with gr.Tab("Setting"):
            with gr.Row():
                model_dd = gr.Dropdown(["gpt-3.5-turbo", "gpt-4-0314", "gpt-4"], value="gpt-3.5-turbo",
                                       label="chat model", info="you can choose the chat model.")
                temperature_sl = gr.Slider(0, 100, 0, step=1, label="temperature (%)")
                max_iterations_sl = gr.Slider(0, 50, 10, step=1, label="max_iterations")

        txt.submit(
            _init, [chatbot, txt], [chatbot, txt]
        ).then(
            clear_text, [], [log_area]
        ).then(bot, [chatbot, model_dd, temperature_sl, max_iterations_sl, ctx_area], [chatbot, ctx_area])

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
