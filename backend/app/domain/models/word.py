from app.component.models.model import Model, Tensor


class EncoderWord(Model):
    def __init__(self) -> None:
        pass

    def forward(self, X: Tensor) -> Tensor:
        return X


class DecoderWord(Model):
    def __init__(self) -> None:
        pass

    def forward(self, X: Tensor) -> Tensor:
        return X
