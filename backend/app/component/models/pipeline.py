from sklearn.pipeline import Pipeline as PipelineSklearn

from .model import Labeller, Model, Tensor


class Pipeline(Model, PipelineSklearn):
    def __init__(self, steps=list[tuple(Model, Labeller)]) -> None:
        super().__init__()

        self.steps = steps

    def forward(self, X: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self).__name__}.forward()")
