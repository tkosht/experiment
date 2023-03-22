import torch
import torch.nn as nn
from typing_extensions import Self

from app.base.models.model import Classifier


class BertClassifier(Classifier):
    def __init__(
        self,
        bert,
        class_names: list[str],
        n_dim=768,
        n_hidden=128,
        n_out=None,
        droprate=0.5,
        weight=None,
    ) -> None:
        super().__init__(class_names)

        self.bert: nn.Module = bert
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.n_out = len(class_names) if n_out is None else n_out
        self.droprate = droprate
        self.weight = weight

        self.criterion = nn.CrossEntropyLoss(weight=weight)

        decoder_layer = nn.TransformerDecoderLayer(d_model=n_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.clf = nn.Sequential(
            nn.BatchNorm1d(self.n_out),
            nn.LogSoftmax(dim=-1),
        )

        self._initialize()

    def _initialize(self) -> Self:
        for p in self.bert.parameters():
            p.requires_grad = False

        for lyr in self.clf.parameters():
            if isinstance(lyr, nn.Linear):
                torch.nn.init.kaiming_uniform_(lyr.weight)

        return self

    def forward(self, *args, **kwargs):
        T = kwargs.pop("target")
        T = torch.transpose(T, 0, 1)  # -> (S', B, V)
        W = self.bert.embeddings.word_embeddings.weight  # (V, D)

        o = self.bert(*args, **kwargs)
        lh = o["last_hidden_state"]
        # po = o["pooler_output"]

        mem = torch.transpose(lh, 0, 1)  # -> (S, B, D)

        shp = list(T.shape)
        shp[0] += 1  # -> (S'+1, B, V)
        tgt = torch.zeros(shp, device=mem.device)
        tgt[1:] = T
        tgt = torch.matmul(tgt, W)  # -> (S'+1, B, D)

        dec = self.decoder(mem, tgt)
        dec = dec[1 : tgt.shape[0]]  # -> (S', B, D)
        assert dec.shape[:-1] == T.shape[:-1]
        dec = torch.transpose(dec, 0, 1)  # -> (B, S', D)
        h = torch.matmul(dec, W.T)  # (B, S', V)

        B, S, V = h.shape
        h = h.reshape(-1, V)  # -> (B*S, V)
        y = self.clf(h).reshape(B, S, V)
        return y

    def loss(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return super().loss(y, t)
