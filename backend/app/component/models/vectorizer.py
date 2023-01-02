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
    def _initialize(self) -> Self:
        self.params = dict(
            vector_size=128,
            sg=1,
            max_vocab_size=1000 * 1000,
            min_count=10,
            window=7,
            epochs=20,
        )
        return self

    def fit(self, X: TextSequences, **params) -> Tensor:
        self.model = word2vec.Word2Vec(X, **self.params)
        return self

    def transform(self, X: TextSequences) -> Tensor:
        # y = [self.model.wv[seq] for seq in X]
        # return y
        return X

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
