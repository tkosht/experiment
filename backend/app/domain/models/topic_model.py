import numpy
from gensim.models import LdaModel
from typing_extensions import Self

from app.component.models.model import Model, Tensor


class TopicModel(Model):
    def __init__(self, n_topic: int = 50, n_epoch: int = 100) -> None:
        self.n_topic = n_topic  # topics
        self.n_epoch = n_epoch  # topics
        self.model = None

        self._initialize()

    def _initialize(self):
        K = self.n_topic
        # NOTE:
        #   alpha: A-priori belief on document-topic distribution
        #   eta: A-priori belief on topic-word distribution
        self.params = dict(
            num_topics=K, iterations=self.n_epoch, alpha=1 / K, eta=1 / K
        )
        # self.params = dict(num_topics=K, iterations=100, alpha="auto", eta="auto")

    def get_topic_probabilities(self, s: slice = slice(None)) -> numpy.ndarray:
        return self.model.expElogbeta[s]

    def fit(self, X: Tensor) -> Self:
        self.model = LdaModel(X, **self.params)
        return self

    def transform(self, X: Tensor, **kwargs) -> Tensor:
        # y = self.model(X)
        # return y
        return X

    def forward(self, X: Tensor) -> Tensor:
        return self.transform(X)

    def __getstate__(self):
        state = super().__getstate__()
        state.update(dict(model=self.model))
        return state

    def __setstate__(self, state):
        model = state.pop("model", None)
        super().__setstate__(state)
        self.model = model
