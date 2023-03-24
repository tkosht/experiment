import torch
import torch.nn as nn
import torch.nn.functional as F
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

        decoder_layer = nn.TransformerDecoderLayer(d_model=n_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.clf = nn.Sequential(
            nn.BatchNorm1d(self.n_out),
            nn.LogSoftmax(dim=-1),
        )

        # loss
        self.context = {}
        self.cel = nn.CrossEntropyLoss(weight=weight)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.mse = nn.MSELoss()

        self._initialize()

    def _initialize(self) -> Self:
        for p in self.bert.parameters():
            p.requires_grad = False

        for lyr in self.parameters():
            if isinstance(lyr, nn.Linear):
                torch.nn.init.kaiming_uniform_(lyr.weight)

        return self

    def create_unk_for(self, mem: torch.Tensor):
        # mem : (S, B, D)
        W = self.bert.embeddings.word_embeddings.weight  # (V, D)
        V, D = W.shape

        # overwrite by unk vector
        S, B, D = mem.shape
        tokenizer = self.context["tokenizer"]
        unk_idx = tokenizer.unk_token_id
        unk = (
            F.one_hot(torch.LongTensor([unk_idx]), num_classes=V)
            .to(torch.float32)
            .reshape(1, 1, -1)
            .repeat(1, B, 1)
            .to(W.device)
        )
        unk_vector = torch.matmul(unk, W)
        U = torch.zeros_like(mem)
        noise_idx = torch.randint(0, S, (1,)).item()
        U[noise_idx] = unk_vector - mem.detach()[noise_idx]

        return U

    def forward(self, *args, **kwargs):
        # TODO:
        #   onehot にノイズ -> UNK に変える / self.tokenizer.unk_token_id
        #   decoder の入力(bert の出力ベクトル)にノイズ (torch.normal(0, 0.1, lh.shape))
        T = kwargs.pop("target")
        T = torch.transpose(T, 0, 1)  # -> (S', B, V)
        W = self.bert.embeddings.word_embeddings.weight  # (V, D)
        self.context["target"] = T

        o = self.bert(*args, **kwargs)
        lh = o["last_hidden_state"]
        # po = o["pooler_output"]

        mem = torch.transpose(lh, 0, 1)  # -> (S, B, D)

        # add unk tensor (more exactly, replace unk vectors)
        U = self.create_unk_for(mem)
        mem = mem + U

        # add noise
        D = lh.shape[-1]
        N = torch.normal(0, 1e-3 / D, mem.shape).to(W.device)
        mem = mem + N

        shp = list(T.shape)
        shp[0] += 1  # -> (S'+1, B, V)
        tgt = torch.zeros(shp, device=mem.device)
        tgt[1:] = T
        tgt = torch.matmul(tgt, W)  # -> (S'+1, B, D)

        dec = self.decoder(mem, tgt)
        dec = dec[1 : tgt.shape[0]]  # -> (S', B, D)
        assert dec.shape[:-1] == T.shape[:-1]
        dec = torch.transpose(dec, 0, 1)  # -> (B, S', D)
        self.context["decoded"] = dec
        h = torch.matmul(dec, W.T)  # (B, S', V)

        B, S, V = h.shape
        h = h.reshape(-1, V)  # -> (B*S, V)
        y = self.clf(h).reshape(B, S, V)
        return y

    def loss(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        loss = super().loss(y, t) + self.loss_difference(y, t) + self.loss_middle()
        return loss

    def loss_middle(self):
        W = self.bert.embeddings.word_embeddings.weight  # (V, D)
        T = self.context["target"]  # -> (S', B, V)
        trg = torch.matmul(T, W)  # -> (S', B, D)

        dec = self.context["decoded"]  # -> (B, S', D)
        dec = torch.transpose(dec, 0, 1)  # -> (S', B, D)
        loss = self.loss_difference(dec, trg)
        return loss

    def loss_difference(self, y: torch.Tensor, t: torch.Tensor):
        return self.mse(y, t) + torch.abs(1 - self.cos(y, t).mean())
