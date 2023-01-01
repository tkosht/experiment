from typing_extensions import Self

from app.component.simple_logger import log_info

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

    def __getitem__(self, item):
        return self.steps[item]

    def get_model(self, idx: int):
        return self.steps[idx][0]

    def get_labeller(self, idx: int):
        return self.steps[idx][1]

    def fit(self, X: Tensor, **kwargs) -> Tensor:
        h = X
        for n_step, (model, labeller) in enumerate(self.steps):
            log_info("Start", "to fit", f"{n_step=}", f"{model=}")
            if labeller is not None:
                assert hasattr(model, "loss")
                t = labeller(h)
                model.fit(h, t, **kwargs)
                continue
            model.fit(h, **kwargs)
            log_info("End", "to fit", f"{n_step=}", f"{model=}")
            log_info("Start", "to transform", f"{n_step=}", f"{model=}")
            h = model.transform(h)
            log_info("End", "to transform", f"{n_step=}", f"{model=}")

    def transform(self, X: Tensor, **kwargs) -> Tensor:
        h = X
        for model, _ in self.steps:
            h = model(h)
        y = h
        return y

    def forward(self, X: Tensor) -> Tensor:
        y = None  # output

        h = X
        for model, labeller in self.steps:
            # apply model
            y = model(h)

            # NOTE: just a simple implement
            if labeller is not None:
                assert hasattr(model, "loss")
                t = labeller(h)
                loss: Tensor = model.loss(y, t)
                print(f"loss={loss.data}")

            # update variable `h` as hidden output tensor
            h = y

        return y
