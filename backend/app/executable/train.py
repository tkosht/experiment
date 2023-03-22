import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

from app.domain.trainer import TrainerBertClassifier
from app.domain.models.simple_models import SimpleBertClassifier


def _main(
    max_epoch: int = 1,
    batch_size: int = 16,
    seed: int = 123456,
    log_interval: int = 10,
    eval_interval: int = 100,
    use_trans: bool = False,
):
    torch.manual_seed(seed)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    dataset = load_dataset("shunk031/JGLUE", name="MARC-ja")

    # setup loader
    trainloader = DataLoader(
        dataset["train"], batch_size=batch_size, num_workers=2, pin_memory=True
    )

    validloader = DataLoader(
        dataset["validation"], batch_size=batch_size, num_workers=2, pin_memory=True
    )

    n_dim = bert.pooler.dense.out_features  # 768
    model = SimpleBertClassifier(
        bert,
        n_dim=n_dim,
        n_hidden=128,  # arbitrary number
        class_names=["differ", "same"],
        weight=torch.Tensor((1 / 9, 1 / 6)),
        use_transdec=use_trans,
    )
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08
    )

    trainer = TrainerBertClassifier(
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        trainloader=trainloader,
        validloader=validloader,
        device=device,
    )

    trainer.do_train(
        max_epoch=max_epoch, log_interval=log_interval, eval_interval=eval_interval
    )


if __name__ == "__main__":
    import typer

    typer.run(_main)
