from gensim.corpora.dictionary import Dictionary
from gensim.models import word2vec
from typing_extensions import Self

from .model import Preprocesser, Tensor, TextSequences


class Vectorizer(Preprocesser):
    def __init__(self) -> None:
        super().__init__()  # must be called at first
        self._initialize()

    def _initialize(self) -> Self:
        return self

    def transform(self, X: TextSequences) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.transform()")

    def fit(self, X: TextSequences) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.fit()")

    def forward(self, X: TextSequences) -> Tensor:
        return self.transform(X)


class VectorizerWord2vec(Vectorizer):
    def __init__(
        self,
        vector_size=128,
        sg=1,
        max_vocab_size=1000 * 1000,
        min_count=10,
        window=7,
        epochs=5,
    ) -> None:
        super().__init__()  # must be called at first

        self.params = dict(
            vector_size=vector_size,
            sg=sg,
            max_vocab_size=max_vocab_size,
            min_count=min_count,
            window=window,
            epochs=epochs,
        )

        self.__dict__.update(self.params)
        self.model = None

    def clear_model(self) -> Self:
        # NOTE: if you would like to train as the first take
        self.model = None
        return self

    def fit(self, X: TextSequences, **kwargs) -> Tensor:
        if self.model is None:
            # the first training
            print("[DEBUG] train firstly")
            self.model = word2vec.Word2Vec(X, **self.params)
        else:
            # the updating training
            print("[DEBUG] train updately")
            self.model.build_vocab(X, update=True)
            self.model.train(
                X, total_examples=self.model.corpus_count, epochs=self.model.epochs
            )
        return self

    def transform(self, X: TextSequences, **kwargs) -> Tensor:
        ws = self.window
        y = []
        for s in X:
            for idw, w in enumerate(s):
                if w in self.model.wv:
                    y.append(self.model.wv[w])
                else:
                    # 辞書にないトークンは、モデルのwindowサイズを前後の文脈語彙として類似ベクトルを推定
                    tokens = [
                        tkn
                        for tkn in s[max(idw - ws, 0) : idw + ws]
                        if tkn in self.model.wv
                    ]
                    try:
                        v = self.model.wv.most_similar(tokens)
                    except Exception as e:
                        raise Exception(
                            f"{e.args[0]} : couldn't get most_similar({tokens=})"
                        )
                    y.append(v)

        return y

    def __getstate__(self):
        state = super().__getstate__()
        state.update(dict(model=self.model))
        return state

    def __setstate__(self, state):
        model = state.pop("model", None)
        super().__setstate__(state)
        self.model = model


class VectorizerBoW(Vectorizer):
    # def __init__(self) -> None:
    #     super().__init__()  # must be called at first
    #     self._initialize()

    def _initialize(self) -> Self:
        self.params = dict()
        return self

    def fit(self, X: TextSequences, **kwargs) -> Tensor:
        self.vocab = Dictionary(X)
        return self

    def transform(self, X: TextSequences) -> Tensor:
        y = [self.vocab.doc2bow(tokens) for tokens in X]
        return y

    def __getstate__(self):
        state = super().__getstate__()
        state.update(dict(vocab=self.vocab))
        return state

    def __setstate__(self, state):
        vocab = state.pop("vocab", None)
        super().__setstate__(state)
        self.vocab = vocab
