import joblib
import json
import typer
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
from app.auto_topic.domain.models.tokenizer import TokenizerWord


from app.auto_topic.component.logger import Logger


g_logger = Logger(logger_name="retrain_wordvector")


def log_info(*args, **kwargs):
    g_logger.info(*args, **kwargs)


def _filter_text(body) -> str:
    soup: BeautifulSoup = BeautifulSoup(body, "lxml")
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


def main(
    batch_size: int = 10000,
    pipe_file: str = "data/pipe_wikivec.gz",
    sentence_file: str = "tidy_sentences.gz",
):
    log_info("Start", "retrain_wordvector")

    with open("data2/bk/2/news.json", "r") as f:
        news = json.load(f)

    cleaned_texts = []
    for itm in news:
        html = itm["html"]
        text = _filter_text(html)
        texts = clean_text(text)
        cleaned_texts.append(texts)

    log_info("Start", "dumping cleand_texts")
    joblib.dump(cleaned_texts, "data/cleaned_texts.gz", compress=("gzip", 3))
    log_info("End", "dumping cleand_texts")

    tokenizer = TokenizerWord(use_stoppoes=False, filterpos=[], use_orgform=False)

    tidy_sentences = []
    for idx, texts in enumerate(tqdm(cleaned_texts)):
        for text in texts:
            tokenized = tokenizer([[text]])
            if not tokenized:
                continue
            snt = tokenized[0]
            if len(snt) < 3:
                continue
            # print(idx, snt)       # for debugging
            tidy_sentences.append("".join(snt))

    log_info("Start", "dumping tidy_sentences")
    joblib.dump(tidy_sentences, "data/tidy_sentences.gz", compress=("gzip", 3))
    log_info("End", "dumping tidy_sentences")

    log_info("End", "retrain_wordvector")


if __name__ == "__main__":
    typer.run(main)
