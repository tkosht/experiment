import os
import random
import traceback as tb
from inspect import signature

import numpy
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score  # , precision_score, recall_score
from tqdm import tqdm

from app.base.component.params import from_config
from app.general.models.trainer import TrainerBertClassifier, g_logger


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_trainer(params: DictConfig) -> TrainerBertClassifier:
    trainer = TrainerBertClassifier()
    trainer.load(load_file=params.trained_file)
    trainer.model.context["tokenizer"] = trainer.tokenizer
    return trainer


def _to_texts(trainer: TrainerBertClassifier, y: torch.Tensor):
    texts = []
    model = trainer.model
    for _y in y:
        text = model.to_text(_y)
        texts.append(text)
    return texts


def do_eval(trainer: TrainerBertClassifier):
    trainer.model.to(trainer.device)
    trainer.model.eval()

    pred_texts = []
    labl_texts = []
    # for bch_idx, bch in enumerate(tqdm(trainer.validloader, desc="evaluating")):
    for bch_idx, bch in enumerate(tqdm(trainer.trainloader, desc="evaluating")):
        inputs, t = trainer._t(bch)
        # y = trainer.model._forward(**inputs)  # for debugging

        with torch.no_grad():
            X = inputs["input_ids"]
            types = inputs["token_type_ids"]
            attn = inputs["attention_mask"]
            p = []
            for _X, _types, _attn in zip(X, types, attn):
                _X = _X.unsqueeze(0)
                _types = _types.unsqueeze(0)
                _attn = _attn.unsqueeze(0)
                ins = dict(input_ids=_X, token_type_ids=_types, attention_mask=_attn)
                _p = trainer.model.predict(**ins)
                p.append(_p)
        pred_texts.extend(p)
        labl_texts.extend(_to_texts(trainer, t))
        if bch_idx >= 10:
            break

    scores = dict(
        acc=accuracy_score(pred_texts, labl_texts),
        # precision=precision_score(pred_texts, labl_texts),
        # recall=recall_score(pred_texts, labl_texts),
    )

    g_logger.info("=" * 80)
    for k, v in scores.items():
        g_logger.info(f"{k}: {v:0.3f}")
    g_logger.info("=" * 80)

    return


def _main(params: DictConfig):
    g_logger.info("Start", "eval")
    g_logger.info("params", f"{params}")

    seed_everything()

    try:
        torch.manual_seed(params.seed)

        trainer = load_trainer(params)

        do_eval(trainer)

    except KeyboardInterrupt:
        g_logger.info("Captured KeyboardInterruption")
    except Exception as e:
        g_logger.error("Error Occured", str(e))
        tb.print_exc()
        raise e
    finally:
        g_logger.info("End", "eval")


@from_config(params_file="conf/app.yml", root_key="/eval")
def config(cfg: DictConfig):
    return cfg


def main(
    seed: int = None,
    trained_file: str = None,  # like "data/trainer.gz"
):
    s = signature(main)
    kwargs = {}
    for k in list(s.parameters):
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    params = config()  # use as default
    params.update(kwargs)
    return _main(params)


if __name__ == "__main__":
    import typer

    typer.run(main)
