from typing_extensions import Self

from .model import Labeller, Model, Tensor


class Pipeline(Model):
    def __init__(self, steps=list[tuple[Model, Labeller]], name="pipeline") -> None:
        super().__init__()

        self.name = name
        self.steps: list[tuple[Model, Labeller]] = steps
        self._initialize()

    def _initialize(self) -> Self:
        for idx, (m, _l) in enumerate(self.steps):
            setattr(self, f"{self.name}_layer_{idx:03d}", m)
        return self

    def forward(self, X: Tensor) -> Tensor:
        y = None  # output

        h = X
        for model, labeller in self.steps:
            # apply model
            y = model(h)

            # NOTE: just a simple implement
            if labeller is not None:
                assert hasattr(model, "loss")
                t = labeller(y)
                loss: Tensor = model.loss(y, t)
                print(f"loss={loss.data}")

            # update variable `h` as hidden output tensor
            h = y

        return y
