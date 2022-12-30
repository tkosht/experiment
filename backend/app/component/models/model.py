from inspect import signature
from typing import TypeAlias

import torch
import torch.nn as nn

Tensor: TypeAlias = torch.FloatTensor
TensorSparse: TypeAlias = Tensor | torch.LongTensor
TensorDense: TypeAlias = Tensor
TensorAny: TypeAlias = TensorDense | TensorSparse

Texts: TypeAlias = list[str]
TextSequences: TypeAlias = list[list[str]]
TensorNumeric: TypeAlias = TensorSparse  # index vector like (9, 3, ..., 2, 1, 2)
TensorOneHot: TypeAlias = TensorSparse  # like (0, ..., 0, 1, 0, ..., 0)
TensorEmbed: TypeAlias = TensorDense


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()  # must be called at first

    def forward(self, X: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.forward()")

    def loss(self, X: Tensor, y: Tensor) -> Tensor:
        return Tensor([0.0])

    def __repr__(self) -> str:
        s = signature(self.__init__)
        params = (f"{k}={self.__dict__[k]}" for k in s.parameters)
        # params = {k: self.__dict__[k] for k in s.parameters}
        return f"{type(self).__name__}({', '.join(params)})"

    def __getstate__(self):
        s = signature(self.__init__)
        state = {}
        for k in list(s.parameters):
            state[k] = self.__dict__[k]
        return state

    def __setstate__(self, state):
        s = signature(self.__init__)
        kwargs = {}
        for k in list(s.parameters):
            kwargs[k] = state[k]
        self.__init__(**kwargs)


class Labeller(Model):
    pass


class Encoder(Model):
    pass


class Decoder(Model):
    def loss(self, X: Tensor, y: Tensor) -> Tensor:
        # NOTE: MUST be implemented
        raise NotImplementedError(f"{type(self).__name__}.loss()")


class Preprocesser(Model):
    pass


class Tokenizer(Model):
    def transform(self, X: Texts) -> TextSequences:
        raise NotImplementedError(f"{type(self).__name__}.transform()")

    def fit(self, X: Texts, y: Texts) -> TextSequences:
        return self

    def forward(self, X: Texts) -> TextSequences:
        return self.transform(X)


class Numericalizer(Model):
    def forward(self, X: TextSequences) -> TensorOneHot | TensorNumeric:
        raise NotImplementedError(f"{type(self).__name__}.forward()")


class Embedder(Model):
    def forward(self, X: TensorOneHot) -> TensorEmbed:
        raise NotImplementedError(f"{type(self).__name__}.forward()")
