import torch.nn.functional as F

from app.auto_topic.component.models.model import (
    Numericalizer,
    TensorNumeric,
    TensorOneHot,
    TextSequences,
)


class NumericalizerSubWord(Numericalizer):
    def __init__(self) -> None:
        # NOTE: SentencePiece モデルを使う
        pass

    def forward(self, X: TextSequences) -> TensorNumeric:
        return X


class ToOneHot(Numericalizer):
    def __init__(self, n_classes: int) -> None:
        # NOTE: SentencePiece モデルを使う
        self.n_classes: int = n_classes

    def forward(self, X: TensorNumeric) -> TensorOneHot:
        return F.one_hot(self.n_classes)
