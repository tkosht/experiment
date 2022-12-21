from app.component.models.model import Numericalizer, TensorOneHot, TextSequences


class NumericalizerSubWord(Numericalizer):
    def __init__(self) -> None:
        # NOTE: SentencePiece モデルを使う
        pass

    def forward(X: TextSequences) -> TensorOneHot:
        return X
