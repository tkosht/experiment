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
        self.params = dict(vector_size=200, min_count=1, window=15, epochs=100)
        return self

    def fit(self, X: TextSequences, **params) -> Tensor:
        self.model = word2vec.Word2Vec(X, **self.params)
        return self

    def transform(self, X: TextSequences) -> Tensor:
        y = [self.model.wv[seq] for seq in X]
        return y

    def __getstate__(self):
        state = super().__getstate__()
        state.update(dict(model=self.model))
        return state

    def __setstate__(self, state):
        model = state.pop("model", None)
        super().__setstate__(state)
        self.model = model