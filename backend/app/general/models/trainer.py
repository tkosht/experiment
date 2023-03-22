import numpy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.base.component.logger import Logger
from app.base.models.model import Classifier
from app.base.models.trainer import TrainerBase

g_logger = Logger(logger_name="simple_trainer")


def log(*args, **kwargs):
    g_logger.info(*args, **kwargs)


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

        for epoch in tqdm(range(max_epoch), desc="epoch"):
            log(f"{epoch=} Start")
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
                    log(f"{epoch=} / {step=}: loss={loss.item():.3f}")

                if step % eval_interval == 0:
                    self.do_eval(max_batches=50, epoch=epoch, step=step)

                if max_batches > 0 and bch_idx >= max_batches:
                    break
            log(f"{epoch=} End")

        log("End training")

    def do_eval(self, max_batches=200, epoch=None, step=None) -> None:
        total_loss = []
        n_corrects = 0
        n_totals = 0

        label_corrects = {}
        labels = {}
        predicts = {}
        predict_corrects = {}

        for bch_idx, bch in enumerate(tqdm(self.validloader, desc="validloader")):
            inputs, T = self._t(bch)

            with torch.no_grad():
                y = self.model(**inputs)
                loss = self.model.loss(y, T)

            total_loss.append(loss.item())

            n_corrects += (y.argmax(dim=-1) == T.argmax(dim=-1)).sum().item()
            bs = len(y)
            n_totals += bs

            def _to_text(tsr: torch.Tensor) -> str:
                return "".join(self.tokenizer.decode(tsr.argmax(dim=-1)))

            for _y, _t in zip(y, T):
                lbl = _to_text(_t)
                prd = _to_text(_y)
                label_corrects[lbl] = label_corrects.get(lbl, 0) + int(prd == lbl)
                labels[lbl] = labels.get(lbl, 0) + 1
                predict_corrects[prd] = predict_corrects.get(prd, 0) + int(prd == lbl)
                predicts[prd] = predicts.get(prd, 0) + 1

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

        # recall
        log("-" * 50)
        for lbl in labels.keys():
            log(
                f"{epoch=} / {step=}: valid recall: {lbl}={label_corrects[lbl] / labels[lbl]:.3f} "
                f"({label_corrects[lbl]} / {labels[lbl]}) "
            )

        # precision
        log("-" * 50)
        for prd in predicts.keys():
            if prd not in labels:
                continue
            log(
                f"{epoch=} / {step=}: valid precision: {prd}={predict_corrects[prd] / predicts[prd]:.3f} "
                f"({predict_corrects[prd]} / {predicts[prd]}) "
            )
        log("=" * 80)
