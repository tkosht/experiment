import torch
import torch.nn as nn

from app.component.models.model import Classifier


class SimpleClassifier(Classifier):
    def __init__(
        self, bert, class_names: list[str], n_dim=768, n_hidden=128, droprate=0.2
    ) -> None:
        super().__init__(class_names)

        self.bert: nn.Module = bert
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.n_classes = len(class_names)
        self.droprate = droprate

        self.clf = nn.Sequential(
            nn.BatchNorm1d(self.n_dim),
            nn.Linear(self.n_dim, self.n_hidden),
            nn.BatchNorm1d(self.n_hidden),
            nn.GELU(),
            nn.Dropout(self.droprate),
            nn.Linear(self.n_hidden, self.n_classes),
            nn.BatchNorm1d(self.n_classes),
            nn.Softmax(dim=-1),
        )

        for p in self.bert.parameters():
            p.requires_grad = False

        for lyr in self.clf:
            if isinstance(lyr, nn.Linear):
                torch.nn.init.kaiming_uniform_(lyr.weight)

    def forward(self, *args, **kwargs):
        o = self.bert(*args, **kwargs)
        lh = o["last_hidden_state"]
        po = o["pooler_output"]
        h = lh.sum(axis=1)
        h = h + po
        y = self.clf(h)

        return y
