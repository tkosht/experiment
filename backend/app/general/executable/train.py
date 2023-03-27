import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingWarmRestarts,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from app.base.component.mlflow_provider import MLFlowProvider
from app.base.component.params import from_config
from app.general.models.model import BertClassifier
from app.general.models.trainer import TrainerBertClassifier, g_logger


def buildup_trainer(params: DictConfig) -> TrainerBertClassifier:
    # resume_file: str,
    # batch_size: int,
    if params.resume_file is not None:
        trainer = TrainerBertClassifier()
        trainer.load(load_file=params.resume_file)
        return trainer

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    dataset: Dataset = load_dataset("shunk031/JGLUE", name="MARC-ja")

    # setup loader
    n_train = (
        params.max_batches * params.batch_size
        if params.max_batches > 0
        else len(dataset["train"])
    )
    trainset = dataset["train"].select(range(n_train))
    trainloader = DataLoader(
        trainset, batch_size=params.batch_size, num_workers=2, pin_memory=True
    )

    validloader = DataLoader(
        dataset["validation"],
        batch_size=params.batch_size,
        num_workers=2,
        pin_memory=True,
    )

    n_dim = bert.pooler.dense.out_features  # 768
    model = BertClassifier(
        bert,
        n_dim=n_dim,
        n_hidden=128,  # arbitrary number
        n_out=tokenizer.vocab_size,
        class_names=["positive", "negative"],
        droprate=0.01,
        weight=None,
    )
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08
    )
    scheduler = ChainedScheduler(
        [
            ConstantLR(optimizer, factor=0.1, total_iters=5),
            CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-4),
        ]
    )

    trainer = TrainerBertClassifier(
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=trainloader,
        validloader=validloader,
        device=device,
    )
    return trainer


@from_config(params_file="conf/app.yml", root_key="/train")
def _main(params: DictConfig):
    g_logger.info("Start", "train")
    g_logger.info("params", f"{params}")

    mlprovider = MLFlowProvider(
        experiment_name="general_trainer",
        run_name="train",
    )

    try:
        torch.manual_seed(params.seed)

        trainer = buildup_trainer(params)
        # resume_file=params.resume_file, batch_size=params.batch_size
        trainer.model.context["tokenizer"] = trainer.tokenizer

        mlprovider.log_params(params)
        mlprovider.log_artifact("conf/app.yml", "conf")

        trainer.do_train(params)
        trainer.save(save_file=params.trained_file)

    except KeyboardInterrupt:
        g_logger.info("Captured KeyboardInterruption")
    except Exception as e:
        g_logger.error("Error Occured", str(e))
        raise e
    finally:
        g_logger.info("End", "train")

        mlprovider.log_metric_from_dict(trainer.metrics)
        mlprovider.log_artifact(params.trained_file, "data")
        mlprovider.log_artifacts(trainer.log_dir, "tb")
        mlprovider.log_artifact("log/app.log", "log")
        mlprovider.end_run()


if __name__ == "__main__":
    _main()
