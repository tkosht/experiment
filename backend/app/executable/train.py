import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

from app.domain.trainer import TrainerClassifier
from app.domain.models.simple_models import SimpleClassifier


# class Tokenizer(object):
#     def __init__(self, tokenizer, device: torch.device = torch.device("cpu")) -> None:
#         self.tokenizer = tokenizer
#         self.device = device
#
#     def __call__(self, bch: dict):
#         sentences = bch["sentence"]
#         labels = bch["label"]
#
#         inputs = self.tokenizer(sentences, return_tensors="pt", padding=True)
#         for k, v in inputs.items():
#             if isinstance(v, torch.Tensor):
#                 inputs[k] = v.to(self.device)
#         t = torch.Tensor(labels).to(self.device)
#         return inputs, t


def _main(
    batch_size: int = 16,
    seed: int = 123456,
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

    model = SimpleClassifier(bert, class_names=["differ", "same"])
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08
    )

    trainer = TrainerClassifier(
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        trainloader=trainloader,
        validloader=validloader,
        device=device,
    )

    trainer.do_train(max_epoch=1, log_interval=10)


if __name__ == "__main__":
    import typer

    typer.run(_main)
