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

from app.component.models.model import Texts, TextSequences, Tokenizer

g_stoppoes = ["BOS/EOS", "助詞", "助動詞", "接続詞", "記号", "補助記号", "未知語"]


# class TokenizerSubWord(Tokenizer):
#     def __init__(self) -> None:
#         # NOTE: SentencePiece モデルを使う
#         pass
#
#     def forward(self, X: Texts) -> TextSequences:
#         raise NotImplementedError(f"{type(self).__name__}.forward()")
#         # return X


class JpTokenizer(Tokenizer):
    def __init__(self, use_stoppoes: bool = False) -> None:
        self.use_stoppoes: bool = use_stoppoes

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
        # return line.split(" ") for example
        raise NotImplementedError("tokenize()")


class JpTokenizerMeCab(JpTokenizer):
    def __init__(self):
        self.dicdir = "/usr/lib/x86_64-linux-gnu/mecab/dic" "/mecab-ipadic-neologd"
        self.taggerstr = f"-O chasen -d {self.dicdir}"
        self.tokenizer = MeCab.Tagger(self.taggerstr)

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
            if self.use_stoppoes and pos not in g_stoppoes:
                sentence.append(word)
        return sentence

    def __getstate__(self):
        state = {
            "dicdir": self.dicdir,
            "taggerstr": self.taggerstr,
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = MeCab.Tagger(self.taggerstr)


class JpTokenizerJanome(JpTokenizer):
    def __init__(self):
        char_filters = [janome.charfilter.UnicodeNormalizeCharFilter()]
        tokenizer = janome.tokenizer.Tokenizer()

        stoppoes = g_stoppoes if self.use_stoppoes else []
        token_filters = [janome.tokenfilter.POSStopFilter(stoppoes)]
        self.aly = janome.analyzer.Analyzer(
            char_filters=char_filters, tokenizer=tokenizer, token_filters=token_filters
        )

    def tokenize(self, line: str) -> Texts:
        sentence = []
        for token in self.aly.analyze(line):
            sentence.append(token.base_form)
        return sentence

    def __getstate__(self):
        state = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__()


class JpTokenizerSudachi(JpTokenizer):
    def __init__(self):
        self.toker = sudachipy.dictionary.Dictionary().create()
        self.mode = sudachipy.tokenizer.Tokenizer.SplitMode.C
        # self.mode = sudachipy.tokenizer.Tokenizer.SplitMode.B

    def tokenize(self, line: str) -> Texts:
        sentence = []
        for token in self.toker.tokenize(line, self.mode):
            if self.use_stoppoes and token.part_of_speech()[0] in g_stoppoes:
                continue
            # sentence.append(token.dictionary_form())
            sentence.append(token.surface())
        return sentence

    def __getstate__(self):
        state = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__()


class JpTokenizerSentencePiece(JpTokenizer):
    def __init__(
        self, input_txt="data/wk/sp.txt", model_prefix="data/wk/sp", vocab_size=2000
    ):

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

    def __getstate__(self):
        state = {
            "input_txt": self.input_txt,
            "model_prefix": self.model_prefix,
            "vocab_size": self.vocab_size,
            "model_file": self.model_file,
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_model()
