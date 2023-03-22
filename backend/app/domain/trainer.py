from inspect import signature

import joblib
import numpy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self

from app.component.logger import Logger
from app.component.models.model import Classifier

g_logger = Logger(logger_name="simple_trainer")


def log(*args, **kwargs):
    g_logger.info(*args, **kwargs)


class TrainerBase(object):
    def do_train(self):
        raise NotImplementedError("do_train()")

    def do_eval(self):
        raise NotImplementedError("do_eval()")

    def __getstate__(self):
        s = signature(self.__init__)
        state = {}
        for k in list(s.parameters):
            state[k] = getattr(self, k)
        return state

    def load(self, load_file: str) -> Self:
        state = joblib.load(load_file)
        self.__init__(**state)
        return self

    def save(self, save_file: str) -> Self:
        s = signature(self.__init__)
        state = {}
        for k in list(s.parameters):
            state[k] = getattr(self, k)
        joblib.dump(state, save_file, compress=("gzip", 3))
        return self


class TrainerBertClassifier(TrainerBase):
    def __init__(
        self,
        tokenizer=None,
        model: Classifier = None,
        optimizer: torch.optim.Optimizer = None,
        trainloader: DataLoader = None,
        validloader: DataLoader = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.validloader = validloader
        self.device = device

    def _t(self, bch: dict) -> None:
        sentences = bch["sentence"]
        labels = bch["label"]

        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        t = torch.Tensor(labels).to(self.device)
        return inputs, t

    def do_train(
        self,
        max_epoch: int = 1,
        max_batches: int = 500,
        log_interval: int = 10,
        eval_interval: int = 100,
    ) -> None:
        log("Start training")
        self.model.to(self.device)

        for epoch in tqdm(range(max_epoch), desc="epoch"):
            log(f"{epoch=} Start")
            for bch_idx, bch in enumerate(tqdm(self.trainloader, desc="trainloader")):
                n_batches = min(max_batches, len(self.trainloader.dataset))
                step = epoch * n_batches + bch_idx
                inputs, t = self._t(bch)

                # train
                self.optimizer.zero_grad()
                y = self.model(**inputs)
                loss = self.model.loss(y, t)
                loss.backward()
                self.optimizer.step()

                if step % log_interval == 0:
                    log(f"{epoch=} / {step=}: loss={loss.item(): .3f}")

                if step % eval_interval == 0:
                    self.do_eval(max_batches=50, epoch=epoch, step=step)

                if max_batches > 0 and bch_idx >= max_batches:
                    break
            log(f"{epoch=} End")

        log("End training")

    def do_eval(self, max_batches=200, epoch=None, step=None) -> None:
        n_classes = self.model.n_classes

        total_loss = []
        n_corrects = 0
        n_totals = 0
        corrects = numpy.zeros(n_classes, dtype=int)
        totals = numpy.zeros(n_classes, dtype=int)
        for bch_idx, bch in enumerate(tqdm(self.validloader, desc="validloader")):
            inputs, t = self._t(bch)

            with torch.no_grad():
                y = self.model(**inputs)
                loss = self.model.loss(y, t)

            total_loss.append(loss.item())

            n_corrects += (y.argmax(dim=-1) == t).sum().item()
            bs = len(y)
            n_totals += bs
            for _y, _t in zip(y, t):
                ldx = _t.item()
                corrects[ldx] += (_y.argmax(dim=-1) == _t).sum().item()  # +0 or +1
                totals[ldx] += 1

            if bch_idx >= max_batches:
                break

        # NOTE: print results
        loss_avg = numpy.array(total_loss).mean()
        log("=" * 80)
        log(f"{epoch=} / {step=}: total valid loss={loss_avg:.3f}")
        log(
            f"{epoch=} / {step=}: total valid accuracy={n_corrects / n_totals:.3f} "
            f"({n_corrects} / {n_totals})"
        )
        log("-" * 50)

        for ldx in range(n_classes):
            lbl = self.model.class_names[ldx]
            log(
                f"{epoch=} / {step=}: valid accuracy={lbl}: {corrects[ldx] / totals[ldx]:.3f} "
                f"({corrects[ldx]} / {totals[ldx]}) "
            )
        log("=" * 80)
