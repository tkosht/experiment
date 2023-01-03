import os
import pathlib
import re

import janome.analyzer
import janome.charfilter
import janome.tokenfilter
import janome.tokenizer
import MeCab
import sentencepiece as spm
import sudachipy.dictionary
import sudachipy.tokenizer
from typing_extensions import Self

from app.component.models.model import Texts, TextSequences, Tokenizer

g_stoppoes = ["BOS/EOS", "助詞", "助動詞", "接続詞", "記号", "補助記号", "未知語"]


class JpTokenizerSubWordBase(Tokenizer):
    def __init__(self) -> None:
        # NOTE: SentencePiece モデルを使う
        pass

    def forward(self, X: Texts) -> TextSequences:
        raise NotImplementedError(f"{type(self).__name__}.forward()")
        # return X


class JpTokenizerWordBase(Tokenizer):
    def __init__(self) -> None:
        # NOTE: MeCab, Janome, Sudachi を使う
        pass

    def forward(self, X: Texts) -> TextSequences:
        raise NotImplementedError(f"{type(self).__name__}.forward()")
        # return X


class JpTokenizer(Tokenizer):
    def _initialize(self) -> Self:
        return self

    def transform(self, X, **kwargs) -> TextSequences:
        docs = []
        for lines in X:
            doc = []
            for line in lines:
                # skip only space line
                if re.search(r"^\s*$", line):
                    continue
                sentence = self.tokenize(line)  # sentence: list
                if len(sentence) <= 0:
                    continue
                doc.extend(sentence)
            if len(doc) <= 0:
                continue
            docs.append(doc)
        return docs

    def tokenize(self, line: str) -> Texts:
        # NOTE:
        # # return line.split(" ") for example
        raise NotImplementedError("tokenize()")


class JpTokenizerMeCab(JpTokenizer):
    def __init__(self, use_stoppoes: bool = False, filterpos=[]) -> None:
        super().__init__()  # must be called at first

        self.use_stoppoes: bool = use_stoppoes
        self.filterpos: list = filterpos

        self.dicdir = "/usr/lib/x86_64-linux-gnu/mecab/dic" "/mecab-ipadic-neologd"
        self.taggerstr = f"-O chasen -d {self.dicdir}"
        self._initialize()

    def _initialize(self) -> Self:
        self.tokenizer = MeCab.Tagger(self.taggerstr)
        return self

    def tokenize(self, line: str) -> Texts:
        sentence = []
        parsed = self.tokenizer.parse(line)
        splitted = [ln.split("\t") for ln in parsed.split("\n")]
        for s in splitted:
            if len(s) == 1:  # may be "EOS"
                break
            word = s[0]  # surface form / form in text
            # word = s[2]  # original form
            pos = s[3].split("-")[0]
            if self.use_stoppoes and pos in g_stoppoes:
                continue
            if self.filterpos and (pos not in self.filterpos):
                continue
            sentence.append(word)
        return sentence


class JpTokenizerJanome(JpTokenizer):
    def __init__(self, use_stoppoes: bool = False) -> None:
        super().__init__()  # must be called at first

        self.use_stoppoes: bool = use_stoppoes
        self._initialize()

    def _initialize(self) -> Self:
        char_filters = [janome.charfilter.UnicodeNormalizeCharFilter()]
        tokenizer = janome.tokenizer.Tokenizer()

        stoppoes = g_stoppoes if self.use_stoppoes else []
        token_filters = [janome.tokenfilter.POSStopFilter(stoppoes)]
        self.aly = janome.analyzer.Analyzer(
            char_filters=char_filters, tokenizer=tokenizer, token_filters=token_filters
        )
        return self

    def tokenize(self, line: str) -> Texts:
        sentence = []
        for token in self.aly.analyze(line):
            sentence.append(token.base_form)
        return sentence


class JpTokenizerSudachi(JpTokenizer):
    def __init__(self, use_stoppoes: bool = False) -> None:
        super().__init__()  # must be called at first

        self.use_stoppoes: bool = use_stoppoes
        self._initialize()

    def _initialize(self) -> Self:
        self.toker = sudachipy.dictionary.Dictionary().create()
        self.mode = sudachipy.tokenizer.Tokenizer.SplitMode.C
        # self.mode = sudachipy.tokenizer.Tokenizer.SplitMode.B
        return self

    def tokenize(self, line: str) -> Texts:
        sentence = []
        for token in self.toker.tokenize(line, self.mode):
            if self.use_stoppoes and token.part_of_speech()[0] in g_stoppoes:
                continue
            # sentence.append(token.dictionary_form())
            sentence.append(token.surface())
        return sentence


class JpTokenizerSentencePiece(JpTokenizer):
    def __init__(
        self, input_txt="data/wk/sp.txt", model_prefix="data/wk/sp", vocab_size=2000
    ):
        super().__init__()  # must be called at first

        self.input_txt = input_txt
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.model_file = f"{self.model_prefix}.model"
        self.sp = None

    def fit(self, X, y, **kwargs):
        pathlib.Path(self.input_txt).parent.mkdir(parents=True, exist_ok=True)
        with open(self.input_txt, "w", encoding="utf-8") as f:
            for doc in X:
                f.writelines(f"{line}{os.linesep}" for line in doc)
        param_str = f"""--input={self.input_txt}
                     --model_prefix={self.model_prefix}
                     --vocab_size={self.vocab_size}
                     """
        param_str = param_str.replace("\n", "")
        spm.SentencePieceTrainer.train(param_str)
        self._load_model()
        return self

    def _load_model(self):
        sp = spm.SentencePieceProcessor()
        sp.Load(self.model_file)
        self.sp = sp

    def tokenize(self, line: str) -> Texts:
        pieces = self.sp.encode_as_pieces(line)
        return pieces


if __name__ == "__main__":
    import joblib

    tokenizer = JpTokenizerJanome(True)
    state = tokenizer.__getstate__()
    joblib.dump(tokenizer, "data/tokenizer.gz", compress=("gzip", 3))
    obj = joblib.load("data/tokenizer.gz")
    repr(obj)
    print(obj)
