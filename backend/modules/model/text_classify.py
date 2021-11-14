import argparse
import os
import pathlib
import re
import sys
import time
from datetime import datetime

import joblib
import lightgbm
from modules.dataset.aozoraset import DatasetAozora
from modules.dataset.ldccset import DatasetLdcc
from sklearn.decomposition import PCA  # , KernelPCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline

try:
    import MeCab
except Exception:
    MeCab = None

try:
    import janome.analyzer
    import janome.charfilter
    import janome.tokenfilter
    import janome.tokenizer
except Exception:
    janome = None

try:
    import sudachipy.dictionary
    import sudachipy.tokenizer
except Exception:
    sudachipy = None

# try:
#     import nagisa
# except Exception:
#     nagisa = None

try:
    import sentencepiece as spm
except Exception:
    spm = None

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class Transer(object):
    def transform(self, X, **kwargs):
        return X

    def fit(self, X, y, **kwargs):
        return self


g_stop_poses = ["BOS/EOS", "助詞", "助動詞", "接続詞", "記号", "補助記号", "未知語"]


class JpTokenizer(Transer):
    def transform(self, X, **kwargs):
        docs = []
        for lines in X:
            doc = []
            for line in lines:
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

    def tokenize(self, line: str) -> list:
        # return line.split(" ") for example
        raise NotImplementedError("tokenize()")


class JpTokenizerMeCab(JpTokenizer):
    def __init__(self):
        self.dicdir = "/usr/lib/x86_64-linux-gnu/mecab/dic" "/mecab-ipadic-neologd"
        self.taggerstr = f"-O chasen -d {self.dicdir}"
        self.tokenizer = MeCab.Tagger(self.taggerstr)

    def tokenize(self, line):
        sentence = []
        parsed = self.tokenizer.parse(line)
        splitted = [ln.split("\t") for ln in parsed.split("\n")]
        for s in splitted:
            if len(s) == 1:  # may be "EOS"
                break
            word = s[0]  # surface form / form in text
            # word = s[2]  # original form
            pos = s[3].split("-")[0]
            if pos not in g_stop_poses:
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
        token_filters = [janome.tokenfilter.POSStopFilter(g_stop_poses)]
        self.aly = janome.analyzer.Analyzer(
            char_filters=char_filters, tokenizer=tokenizer, token_filters=token_filters
        )

    def tokenize(self, line):
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

    def tokenize(self, line):
        sentence = []
        for token in self.toker.tokenize(line, self.mode):
            if token.part_of_speech()[0] not in g_stop_poses:
                # sentence.append(token.dictionary_form())
                sentence.append(token.surface())
        return sentence

    def __getstate__(self):
        state = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__init__()


class JpTokenizerNagisa(JpTokenizer):
    def tokenize(self, line):
        tagged = nagisa.filter(line, filter_postags=g_stop_poses)
        return tagged.words


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

    def tokenize(self, line):
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


class Splitter(Transer):
    def __init__(self, sep=" "):
        self.sep = sep

    def transform(self, X: list, **kwargs):
        return [x.split(self.sep) for x in X]


class SparsetoDense(Transer):
    def transform(self, X, **kwargs):
        return X.toarray()


class TagDocMaker(Transer):
    def transform(self, X, **kwargs):
        return [
            TaggedDocument(words=sentences, tags=[n]) for n, sentences in enumerate(X)
        ]


class Doc2Vectorizer(Transer):
    def __init__(self, n_components, window=7, min_count=1, workers=3):
        super().__init__()
        self.model = None
        self.vector_size = n_components
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def transform(self, X, **kwargs):
        embedded = []
        for tagdoc in X:
            v = self.model.infer_vector(tagdoc.words)
            embedded.append(v)
        return embedded

    def fit(self, X, y, **kwargs):
        assert isinstance(X, list)
        assert isinstance(X[0], TaggedDocument)
        params = dict(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            dm=0,
            dbow_words=0,
            negative=0,
            hs=1,
        )
        self.model = Doc2Vec(X, **params)
        return self


def ident_tokener(sentence):
    return sentence


def build_pipleline_simple(tokener):
    tfidf = TfidfVectorizer(tokenizer=ident_tokener, lowercase=False)

    embedders = [
        ("pca", PCA(n_components=16)),
        ("identity", Transer()),  # means tfidf to tfidf
    ]

    lgbmclf = lightgbm.LGBMClassifier(
        objective="softmax",
        num_class=len(dataset.labelset),
        importance_type="gain",
    )

    pipe = Pipeline(
        steps=[
            ("tokenizer", tokener),
            ("vectorizer", tfidf),
            ("to_dence", SparsetoDense()),
            ("embedder", FeatureUnion(embedders)),
            ("classifier", lgbmclf),
        ]
    )

    return pipe


def build_pipleline_with_doc2vec(tokener):
    tfidf = TfidfVectorizer(tokenizer=ident_tokener, lowercase=False)

    embedders = [
        ("pca", PCA(n_components=32)),
        ("identity", Transer()),  # means tfidf to tfidf
    ]

    pipe_embedder_1 = Pipeline(
        steps=[
            ("vectorizer", tfidf),
            ("to_dence", SparsetoDense()),
            ("embedder", FeatureUnion(embedders)),
        ]
    )
    pipe_embedder_2 = Pipeline(
        steps=[
            ("doctagger", TagDocMaker()),
            ("doc2vec", Doc2Vectorizer(n_components=128, min_count=1)),
        ]
    )
    pipe_embeds = [
        ("pipe1", pipe_embedder_1),
        ("pipe2", pipe_embedder_2),
    ]

    lgbmclf = lightgbm.LGBMClassifier(
        objective="softmax",
        num_class=len(dataset.labelset),
        importance_type="gain",
    )

    pipe = Pipeline(
        steps=[
            ("tokenizer", tokener),
            ("embedders", FeatureUnion(pipe_embeds)),
            ("classifier", lgbmclf),
        ]
    )

    return pipe


def get_args():
    parser = argparse.ArgumentParser(description="VAE MNIST Example")
    parser.add_argument(
        "--dataset",
        choices=["ldcc", "aozora"],
        default="aozora",
        help='string for the dataset name (default: "aozora")',
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=3,
        help="positive integer of the iteration "
        "for train and validation (default: 10)",
    )
    args = parser.parse_args()
    return args


def get_now():
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S")


def print_log(*params, **kwparams):
    print(get_now(), *params, **kwparams)


def _do_train(dataset, tokenizers):
    # loop to train and validation
    print("datetime, tokenizer, train_acc, valid_acc, elapsed_time, cpu_time")
    for n_iter in range(args.iter):
        dataset.shuffle().split()
        X_train, X_valid = dataset.get_data(do_split=True)
        y_train, y_valid = dataset.get_labels(do_split=True)

        data_file = f"data/dataset/{args.dataset}set_iter{n_iter:02d}.gz"
        print_log(f"Saving dataset ... [{data_file}]")
        joblib.dump(dataset, data_file, compress=("gzip", 3))

        for tkr in tokenizers:
            print_log(tkr.__class__.__name__, "Processing ...", file=sys.stderr)

            pipe = build_pipleline_simple(tkr)
            # pipe = build_pipleline_with_doc2vec(tkr)

            tps = time.perf_counter()
            tcs = time.process_time()

            # train with trainset
            pipe.fit(X_train, y_train)

            # predict trainset
            p = pipe.predict(X_train)
            train_acc = accuracy_score(y_train, p)

            # predict validset
            p = pipe.predict(X_valid)
            valid_acc = accuracy_score(y_valid, p)

            tce = time.process_time()
            tpe = time.perf_counter()

            elapsed_time = tpe - tps
            cpu_time = tce - tcs

            print_log(
                ",",
                f"{tkr.__class__.__name__}, "
                f"{train_acc}, {valid_acc}, "
                f"{elapsed_time}, {cpu_time}",
            )

            # save model
            print_log(
                f"Saving model for {tkr.__class__.__name__} at iter {n_iter:02d} ..."
            )
            pipe_file = f"data/model/pipe-{tkr.__class__.__name__.lower()}_{args.dataset}set_iter{n_iter:02d}.gz"
            joblib.dump(pipe, pipe_file, compress=("gzip", 3))

            print_log(tkr.__class__.__name__, "Done.", file=sys.stderr)


if __name__ == "__main__":
    args = get_args()

    # load dataset
    data_file = f"data/dataset/{args.dataset}set.gz"
    if pathlib.Path(data_file).exists():
        dataset = joblib.load(data_file)
    else:
        print_log(f"loading dataset {args.dataset} ...")
        dataset_class_dic = dict(
            ldcc=DatasetLdcc,
            aozora=DatasetAozora,
        )

        dataset_class = dataset_class_dic[args.dataset]
        dataset = dataset_class()
        dataset.load()
        print_log(f"loading dataset {args.dataset} ... Done.")

        print_log(f"Saving dataset ... [{data_file}]")
        pathlib.Path(data_file).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dataset, data_file, compress=("gzip", 3))

    print_log(f"using dataset: {args.dataset}")

    # setup tokenizers
    tokenizers = []
    if MeCab is not None:
        tokenizers.append(JpTokenizerMeCab())
    if janome is not None:
        tokenizers.append(JpTokenizerJanome())
    if sudachipy is not None:
        tokenizers.append(JpTokenizerSudachi())
    # if nagisa is not None:    # too slow at version 0.2.7, 2021/10
    #     tokenizers.append(JpTokenizerNagisa())
    if spm is not None:
        tokenizers.append(JpTokenizerSentencePiece(vocab_size=5000))

    _do_train(dataset, tokenizers)
