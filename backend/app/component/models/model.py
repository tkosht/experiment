from typing import TypeAlias

import torch
import torch.nn as nn

Tensor: TypeAlias = torch.FloatTensor
TensorSparse: TypeAlias = Tensor | torch.LongTensor
TensorDense: TypeAlias = Tensor
TensorAny: TypeAlias = TensorDense | TensorSparse

Texts: TypeAlias = list[str]
TextSequences: TypeAlias = list[list[str]]
TensorOneHot: TypeAlias = TensorSparse
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


class Tokenizer(Model):
    def forward(self, X: Texts) -> TextSequences:
        raise NotImplementedError(f"{type(self).__name__}.forward()")


class Numericalizer(Model):
    def forward(self, X: TextSequences) -> TensorOneHot:
        raise NotImplementedError(f"{type(self).__name__}.forward()")


class Embedder(Model):
    def forward(self, X: TensorOneHot) -> TensorEmbed:
        raise NotImplementedError(f"{type(self).__name__}.forward()")