from app.component.models.model import Texts, TextSequences, Tokenizer


class TokenizerSubWord(Tokenizer):
    def __init__(self) -> None:
        # NOTE: SentencePiece モデルを使う
        pass

    def forward(X: Texts) -> TextSequences:
        return X
