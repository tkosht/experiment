import json
import re

import joblib
import typer
from bs4 import BeautifulSoup
from tqdm import tqdm

from app.auto_topic.component.logger import Logger
from app.auto_topic.domain.models.tokenizer import TokenizerWord

g_logger = Logger(logger_name="retrain_wordvector")


def log_info(*args, **kwargs):
    g_logger.info(*args, **kwargs)


def _filter_text(body) -> str:
    soup: BeautifulSoup = BeautifulSoup(body, "lxml-xml")
    for tg in ["script", "noscript", "meta"]:
        try:
            soup.find(tg).replace_with(" ")
        except Exception:
            # NOTE: Not Found `tg` tag
            pass
    return soup.get_text("\n\n")


def clean_text(text: str):
    contents = []
    for txt in re.split(r"(。|\n)", text):
        txt = txt.strip().replace("\u200b", "").replace("\u3000", " ")
        txt = re.sub(r"\n+", "\n", txt)
        txt = re.sub(r"([\W])\1+", " ", txt)
        if not txt:
            continue
        # contents.append(txt)
        # contents.extend(txt.split("\n"))
        contents.append(txt.split("\n")[-1])
    return contents


def make_sentences(json_file: str):
    with open(json_file, "r") as f:
        news = json.load(f)

    cleaned_texts = []
    for itm in news:
        html = itm["html"]
        text = _filter_text(html)
        texts = clean_text(text)
        cleaned_texts.append(texts)

    tokenizer = TokenizerWord(use_stoppoes=False, filterpos=[], use_orgform=False)
    tidy_sentences = []
    for idx, texts in enumerate(cleaned_texts):
        for text in texts:
            tokenized = tokenizer([[text]])
            if not tokenized:
                continue
            snt = tokenized[0]
            if len(snt) < 3:
                continue
            # print(idx, snt)       # for debugging
            tidy_sentences.append("".join(snt))
    return tidy_sentences


def main(
    batch_size: int = 10000,
    base_pipe_file: str = "data/pipe_wikivec.gz",
    news_json_file: str = "app/scrapy/data/news.json",
):
    log_info("Start", "retrain_wordvector")
    log_info("Start", "loading trained vectors")
    pipe_wikivec = joblib.load(base_pipe_file)
    log_info("End", "loading trained vectors")

    log_info("Start", "make sentences")
    tidy_sentences = make_sentences(news_json_file)
    log_info("End", "make sentences")

    log_info("Start", "tune model")
    for idx, offset in enumerate(tqdm(range(0, len(tidy_sentences), batch_size))):
        X = tidy_sentences[offset : offset + batch_size]
        print(offset, offset + batch_size)
        pipe_wikivec.fit([X])
    log_info("End", "tune model")

    # NOTE: around 2m = 1m55.0s
    log_info("Start", "dumping retrained model")
    joblib.dump(pipe_wikivec, "data/pipe_wikivec.tuned.gz", compress=("gzip", 3))
    log_info("End", "dumping retrained model")

    log_info("End", "retrain_wordvector")


if __name__ == "__main__":
    typer.run(main)
