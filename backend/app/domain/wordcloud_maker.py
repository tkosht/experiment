from __future__ import annotations

import collections
import pathlib
import re
import traceback as tb

import requests
from bs4 import BeautifulSoup
from janome.analyzer import Analyzer
from janome.charfilter import RegexReplaceCharFilter, UnicodeNormalizeCharFilter
from janome.tokenfilter import ExtractAttributeFilter, LowerCaseFilter, POSKeepFilter
from janome.tokenizer import Tokenizer
from wordcloud import WordCloud


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


class WordCloudMaker(object):
    font_path = "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf"

    def __init__(self, url: str) -> None:
        self.url = url

        self.tokenizer = Tokenizer()
        self.char_filters = [
            UnicodeNormalizeCharFilter(),
            RegexReplaceCharFilter("<.*?>", ""),
        ]

        self.token_filters = [
            POSKeepFilter(["名詞"]),
            LowerCaseFilter(),
            ExtractAttributeFilter("base_form"),
        ]

        self.alz = Analyzer(
            char_filters=self.char_filters,
            tokenizer=self.tokenizer,
            token_filters=self.token_filters,
        )

        self.params = dict(
            background_color="white",
            font_path=self.font_path,
            width=600,
            height=400,
            min_font_size=15,
        )
        self.wc = WordCloud(**self.params)
        self.dir_name = "data/wc"

    def clean_text(self, text: str):
        sentences = []
        sents = text.split("。")
        for s in sents:
            poses = [
                tkn.part_of_speech.split(",")[0] for tkn in self.tokenizer.tokenize(s)
            ]
            if "動詞" not in poses:
                continue
            sentences.append(s)

        return "。".join(sentences)

    def count_words(self, text: str) -> dict:
        words = collections.Counter(self.alz.analyze(text))
        return words

    def make(self, file_id="wordcloud") -> WordCloudMaker:
        dwl = Downloader()
        text = dwl.download(self.url)

        soup = BeautifulSoup(text, "html.parser")
        text = re.sub(r"\n+", "\n", soup.text)
        cleaned = self.clean_text(text)
        words = self.count_words(cleaned)

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
