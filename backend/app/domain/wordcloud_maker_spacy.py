from __future__ import annotations

import pathlib
import re
import traceback as tb

import requests
import spacy
from bs4 import BeautifulSoup
from wordcloud import WordCloud

g_nlp = spacy.load('ja_ginza_electra')


class Downloader(object):
    def __init__(self) -> None:
        pass

    def download(self, url) -> str:
        response = requests.get(url, allow_redirects=True)
        if response.status_code != 200:
            raise Exception(f"HTTP status error {response.status_code=}")

        # assert response.content.decode(response.encoding) == response.text
        text = response.content.decode(response.apparent_encoding) 
        return text


def clean_text(text: str):
    doc = g_nlp(text)

    sentences = []
    for sents in doc.sents:
        poses = [tkn.pos_ for tkn in sents]
        if "VERB" not in poses:
            continue
        sentences.append(sents.text.strip())
    
    return "".join(sentences)


def count_words(text: str, poses=["NOUN"]) -> dict:
    doc = g_nlp(text)
    
    words = {}
    for sents in doc.sents:
        for tkn in sents:
            if tkn.pos_ not in poses:
                continue
            if tkn.lemma_.strip() == "":
                continue
            cnt = words.get(tkn.lemma_, 0)
            words[tkn.lemma_] = cnt + 1
    return words


class WordCloudMaker(object):
    font_path = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"

    def __init__(self, url: str) -> None:
        self.url = url
        self.params = dict(
            background_color="white",
            font_path=self.font_path,
            width=600,
            height=400,
            min_font_size=15
        )
        self.wc = WordCloud(**self.params)
        self.dir_name = "data/wc"

    def make(self, file_id="wordcloud") -> WordCloudMaker:
        dwl = Downloader()
        text = dwl.download(self.url)

        soup = BeautifulSoup(text, 'html.parser')
        text = re.sub(r"\n+", "\n", soup.text)
        cleaned = clean_text(text)
        words = count_words(cleaned, poses=["NOUN"])

        self.wc.generate_from_frequencies(words)

        pathlib.Path(self.dir_name).mkdir(parents=True, exist_ok=True)
        self.wc.to_file(f"data/wc/{file_id}.png")
        return self


def _main():
    import ulid
    try:
        # url: str = "https://www.jiji.com/jc/article?k=2022072000771&g=soc"
        url: str = "https://www3.nhk.or.jp/news/html/20220720/k10013728301000.html"
        file_id: str = str(ulid.new())

        wcm = WordCloudMaker(url)
        wcm.make(file_id)
        print(f"saved {file_id=}")
    except Exception as e:
        print(f"Error: {e}", tb.format_exc())
    finally:
        print("Done.")


if __name__ == "__main__":
    import typer
    typer.run(_main)
