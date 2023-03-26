import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ChainedScheduler,
    ConstantLR,
)
from app.general.models.model import BertClassifier
from app.general.models.trainer import TrainerBase, TrainerBertClassifier, g_logger


def buildup_trainer(
    resume_file: str,
    batch_size: int,
) -> TrainerBase:
    if resume_file is not None:
        trainer = TrainerBertClassifier()
        trainer.load(load_file=resume_file)
        return trainer

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    dataset: Dataset = load_dataset("shunk031/JGLUE", name="MARC-ja")

    # setup loader
    trainloader = DataLoader(
        dataset["train"], batch_size=batch_size, num_workers=2, pin_memory=True
    )

    validloader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
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


def _main(
    max_epoch: int = 1,
    max_batches: int = 1,
    batch_size: int = 16,
    seed: int = 123456,
    log_interval: int = 10,
    eval_interval: int = 100,
    resume_file: str = None,  # like "data/trainer.gz"
    trained_file: str = "data/trainer.gz",
):
    torch.manual_seed(seed)

    g_logger.info("Start", "train")

    g_logger.info(f"{max_epoch=}")
    g_logger.info(f"{max_batches=}")
    g_logger.info(f"{batch_size=}")
    g_logger.info(f"{seed=}")
    g_logger.info(f"{log_interval=}")
    g_logger.info(f"{eval_interval=}")
    g_logger.info(f"{resume_file=}")
    g_logger.info(f"{trained_file=}")

    try:
        trainer = buildup_trainer(resume_file=resume_file, batch_size=batch_size)
        trainer.model.context["tokenizer"] = trainer.tokenizer

        trainer.do_train(
            max_epoch=max_epoch,
            max_batches=max_batches,
            log_interval=log_interval,
            eval_interval=eval_interval,
        )

        trainer.save(save_file=trained_file)
    except KeyboardInterrupt:
        g_logger.info("Captured KeyboardInterruption")
    except Exception as e:
        g_logger.error("Error Occured", str(e))
        raise e
    finally:
        g_logger.info("End", "train")


if __name__ == "__main__":
    import typer

    typer.run(_main)
