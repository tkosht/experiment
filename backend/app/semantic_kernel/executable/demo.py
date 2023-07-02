import re
from inspect import signature

import gradio as gr
import typer
from omegaconf import DictConfig

from app.semantic_kernel.component.semantic_bot import SemanticBot


def _init(history: list[tuple[str, str]], text: str):
    history = history + [(text, None)]
    return history, "", ""


def _find_urls(text_contains_urls: str) -> list[str]:
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    urls = re.findall(url_pattern, text_contains_urls)
    return urls


class BotWrapper(object):
    def __init__(self, bot: SemanticBot, memory_chunk_size: int = 1024) -> None:
        self.bot = bot
        self.memory_chunk_size = memory_chunk_size

        self.processed = set()

    async def _add_memory(self, text: str):
        import requests
        from bs4 import BeautifulSoup

        urls = _find_urls(text)

        successful_urls = []
        for url in urls:
            if url in self.processed:
                continue

            try:
                self.processed.add(url)

                response = requests.get(url)
                soup = BeautifulSoup(response.text, "lxml")

                for idx in range(0, len(soup.text), self.memory_chunk_size):
                    await self.bot.memory.append(
                        text=soup.text[idx : idx + self.memory_chunk_size],
                        additional_metadata=url,
                    )
                successful_urls.append(url)
            except Exception as e:
                print(e)
                continue

        if not successful_urls:
            return "no urls to be loaded."

        return "success to loaded to the memory: \n" + "\n".join(successful_urls)


def _main(params: DictConfig):
    default_query = "LLMについて教えて？"

    with gr.Blocks() as demo:
        bot = SemanticBot()
        bot_wrapper = BotWrapper(bot=bot)

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
                with gr.Column():
                    url_txt = gr.Textbox(
                        show_label=True,
                        label="please input url ",
                        placeholder="No Memory",
                        value="https://xtech.nikkei.com/atcl/nxt/column/18/02504/062600003/",
                    ).style(container=False)
                    btn = gr.Button(value="Add Memory")

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
        btn.click(bot_wrapper._add_memory, [url_txt], [url_txt])

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
