import numpy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.component.logger import Logger
from app.component.models.model import Classifier

g_logger = Logger(logger_name="simple_trainer")


def log(*args, **kwargs):
    g_logger.info(*args, **kwargs)


class TrainerClassifier(object):
    def __init__(
        self,
        tokenizer,
        model: Classifier,
        optimizer: torch.optim.Optimizer,
        trainloader: DataLoader,
        validloader: DataLoader,
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
        max_step=500,
        log_interval: int = 10,
        valid_interval: int = 100,
    ) -> None:
        log("Start training")
        self.model.to(self.device)

        for epoch in range(max_epoch):
            for bch_idx, bch in enumerate(tqdm(self.trainloader, desc="train")):
                step = epoch * len(self.trainloader.dataset) + bch_idx
                inputs, t = self._t(bch)

                # train
                self.optimizer.zero_grad()
                y = self.model(**inputs)
                loss = self.model.loss(y, t)
                loss.backward()
                self.optimizer.step()

                if step % log_interval == 0:
                    log(f"{epoch=} / {step=}: loss={loss.item(): .3f}")

                if step % valid_interval == 0:
                    self.do_eval(max_step=50)

                if step >= max_step:
                    break
        log("End training")

    def do_eval(self, max_step=50) -> None:
        n_classes = len(self.model.class_names)

        total_loss = []
        n_corrects = 0
        n_totals = 0
        corrects = numpy.zeros(n_classes, dtype=int)
        totals = numpy.zeros(n_classes, dtype=int)
        for bch_idx, bch in enumerate(tqdm(self.trainloader, desc="validation")):
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

            if bch_idx >= max_step:
                break

        # NOTE: print results
        loss_avg = numpy.array(total_loss).mean()
        log("=" * 80)
        log(f"valid loss={loss_avg:.3f}")
        log(f"total accuracy={n_corrects / n_totals:.3f} ({n_corrects} / {n_totals})")
        log("-" * 50)

        for ldx in range(n_classes):
            lbl = self.model.class_names[ldx]
            log(
                f"{lbl}: {corrects[ldx] / totals[ldx]:.3f} ({corrects[ldx]} / {totals[ldx]})"
            )
        log("=" * 80)
