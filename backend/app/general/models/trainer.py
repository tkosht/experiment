import numpy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self

from app.base.component.logger import Logger
from app.base.models.model import Classifier
from app.base.models.trainer import TrainerBase

g_logger = Logger(logger_name="general_trainer")


def log(*args, **kwargs):
    g_logger.info(*args, **kwargs)


class Score(object):
    def __init__(self, tokenizer) -> None:
        self.Ps: list = []
        self.Ts: list = []

        self.tokenizer = tokenizer

        self.n_corrects = 0
        self.n_totals = 0
        self.label_corrects = {}
        self.labels = {}
        self.predicts = {}
        self.predict_corrects = {}

    def append(self, P: torch.Tensor, T: torch.Tensor) -> Self:
        self.Ps.append(P)
        self.Ts.append(T)
        return self

    def _to_text(self, tsr: torch.Tensor) -> str:
        return "".join(self.tokenizer.decode(tsr.argmax(dim=-1)))

    def calculate(self):
        for _P, _T in zip(self.Ps, self.Ts):
            for p, t in zip(_P, _T):
                lbl = self._to_text(t)
                prd = self._to_text(p)
                self.label_corrects[lbl] = self.label_corrects.get(lbl, 0) + int(
                    prd == lbl
                )
                self.labels[lbl] = self.labels.get(lbl, 0) + 1
                self.predict_corrects[prd] = self.predict_corrects.get(prd, 0) + int(
                    prd == lbl
                )
                self.predicts[prd] = self.predicts.get(prd, 0) + 1
                self.n_corrects += int(prd == lbl)
                self.n_totals += 1


class TrainerBertClassifier(TrainerBase):
    def __init__(
        self,
        tokenizer=None,
        model: Classifier = None,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        trainloader: DataLoader = None,
        validloader: DataLoader = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.validloader = validloader
        self.device = device

    def _to_device(self, d: dict) -> dict:
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(self.device)
        return d

    def _t(self, bch: dict) -> None:
        sentences = bch["sentence"]
        labels = [self.model.class_names[ldx] for ldx in bch["label"]]

        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)
        self._to_device(inputs)

        teachers = self.tokenizer(labels, return_tensors="pt", padding=True)
        self._to_device(teachers)
        t = teachers.input_ids

        T = F.one_hot(t, num_classes=self.tokenizer.vocab_size)
        T = T.to(torch.float32)
        inputs["target"] = T

        return inputs, T

    def do_train(
        self,
        max_epoch: int = 1,
        max_batches: int = 500,
        log_interval: int = 10,
        eval_interval: int = 100,
    ) -> None:
        log("Start training")
        self.model.to(self.device)
        n_batches = len(self.trainloader)

        step = 0
        for epoch in tqdm(range(max_epoch), desc="epoch"):
            log(f"{epoch=} Start")
            for lrx, lr in enumerate(self.scheduler.get_last_lr()):
                self.write_board(f"10.learnig_rate/{lrx:02d}", lr, step)

            for bch_idx, bch in enumerate(tqdm(self.trainloader, desc="trainloader")):
                n_batches = min(max_batches, len(self.trainloader.dataset))
                step = epoch * n_batches + bch_idx
                inputs, T = self._t(bch)

                # train
                self.optimizer.zero_grad()
                y = self.model(**inputs)
                loss = self.model.loss(y, T)
                loss.backward()
                self.optimizer.step()

                if step % log_interval == 0:
                    log(f"{epoch=} / {step=}: loss={loss.item():.7f}")
                    self.write_board("01.loss/train", loss.item(), step)

                    score = Score(self.tokenizer).append(y.detach(), T.detach())
                    loss_train = loss.item()
                    self.log_loss("train", loss_train, epoch, step)
                    self.log_scores("train", score, epoch, step)

                if step % eval_interval == 0:
                    self.do_eval(max_batches=50, epoch=epoch, step=step)

                if max_batches > 0 and bch_idx >= max_batches:
                    break
            log(f"{epoch=} End")

            self.scheduler.step()

        log("End training")

    def do_eval(self, max_batches=200, epoch=None, step=None) -> None:
        total_loss = []
        score = Score(self.tokenizer)
        for bch_idx, bch in enumerate(tqdm(self.validloader, desc="validloader")):
            inputs, T = self._t(bch)

            with torch.no_grad():
                y = self.model(**inputs)
                loss = self.model.loss(y, T)

            total_loss.append(loss.item())
            score.append(y, T)

            if bch_idx >= max_batches:
                break

        loss_valid = numpy.array(total_loss).mean()
        self.log_loss("valid", loss_valid, epoch, step)
        self.log_scores("valid", score, epoch, step)

    def log_loss(
        self, key: str, loss_value: float, epoch: int = None, step: int = None
    ):
        log("=" * 80)
        log(f"{epoch=} / {step=}: {key} loss={loss_value:.7f}")
        self.write_board(f"01.loss/{key}", loss_value, step)

    def log_scores(self, key: str, score: Score, epoch: int = None, step: int = None):
        score.calculate()

        # accuracy
        n_corrects = score.n_corrects
        n_totals = score.n_totals
        log(
            f"{epoch=} / {step=}: total {key} accuracy={n_corrects / n_totals:.3f} "
            f"({n_corrects} / {n_totals})"
        )
        self.write_board(f"02.accuracy/{key}", n_corrects / n_totals, step)

        # recall
        log("-" * 50)
        for lbl in score.labels.keys():
            log(
                f"{epoch=} / {step=}: {key} recall: {lbl}={score.label_corrects[lbl] / score.labels[lbl]:.3f} "
                f"({score.label_corrects[lbl]} / {score.labels[lbl]}) "
            )
            self.write_board(
                f"03.recall/{key}/{lbl}",
                score.label_corrects[lbl] / score.labels[lbl],
                step,
            )
            # setup fake info
            if lbl not in score.predicts:
                score.predicts[lbl] = -1
                score.predict_corrects[lbl] = 0

        # precision
        log("-" * 50)
        n_predicts = n_predict_corrects = 0
        for prd in score.predicts.keys():
            if prd not in score.labels:
                n_predicts += score.predicts[prd]
                n_predict_corrects += score.predict_corrects[prd]
                # continue
            log(
                f"{epoch=} / {step=}: {key} precision: {prd}={score.predict_corrects[prd] / score.predicts[prd]:.3f} "
                f"({score.predict_corrects[prd]} / {score.predicts[prd]}) "
            )
            self.write_board(
                f"04.precision/{key}/{lbl}",
                score.predict_corrects[prd] / score.predicts[prd],
                step,
            )
        log(
            f"{epoch=} / {step=}: {key} precision: others={n_predict_corrects / n_predicts:.3f} "
            f"({n_predict_corrects} / {n_predicts}) "
        )
        self.write_board(
            f"04.precision/{key}/others", n_predict_corrects / n_predicts, step
        )
        log("=" * 80)

    def load(self, load_file: str) -> Self:
        log(f"Loading trainer ... [{load_file}]")
        me = super().load(load_file)
        log(f"Loaded trainer ... [{load_file}]")
        return me

    def save(self, save_file: str) -> Self:
        log(f"Saving trainer ... [{save_file}]")
        super().save(save_file)
        log(f"Saved trainer ... [{save_file}]")
        return self
