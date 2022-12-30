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
    def forward(self, X: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.forward()")

    def loss(self, X: Tensor, y: Tensor) -> Tensor:
        return Tensor([0.0])


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
