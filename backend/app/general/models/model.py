import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self

from app.base.component.params import add_args
from app.base.models.model import Classifier  # , Reshaper


class BertClassifier(Classifier):
    @add_args(params_file="conf/app.yml", root_key="/model/transformer/decoder")
    def __init__(
        self,
        bert,
        class_names: list[str],
        n_dim=768,
        n_hidden=128,
        n_out=None,
        droprate=0.5,
        weight=None,
        nhead=8,
        num_layers=2,
    ) -> None:
        super().__init__(class_names)

        self.bert: nn.Module = bert
        self.n_dim = n_dim
        self.n_hidden = n_hidden
        self.n_out = len(class_names) if n_out is None else n_out
        self.droprate = droprate
        self.weight = weight
        self.nhead = nhead
        self.num_layers = num_layers

        decoder_layer = nn.TransformerDecoderLayer(d_model=n_dim, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.clf = nn.Sequential(
            # nn.Linear(self.n_dim, self.n_out),
            # Reshaper(shp=(-1, self.n_out)),
            nn.BatchNorm1d(self.n_out),
            nn.LogSoftmax(dim=-1),
            # nn.Softmax(dim=-1),
        )

        # loss
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

    def create_right_shift_target(self, T: torch.Tensor):
        shp = list(T.shape)
        shp[0] += 1  # -> (S'+1, B, D)
        tgt = torch.zeros(shp, device=T.device)
        tgt[1:] = T
        return tgt

    def forward(self, *args, **kwargs):
        o = self.bert(*args, **kwargs)
        mem = torch.transpose(o["last_hidden_state"], 0, 1)  # -> (S, B, D)
        # po = o["pooler_output"]

        teachers = self.context["teachers"]
        to = self.bert(**teachers)
        trg = to["last_hidden_state"]
        T = torch.transpose(trg, 0, 1)  # -> (S', B, D)
        self.context["target"] = trg  # -> (B, S', D)

        # # NOTE: mem に揺らぎを与える
        # #   - onehot を UNKnown に変える / self.tokenizer.unk_token_id
        # #   - decoder の入力(bert の出力ベクトル)にノイズ (torch.normal(0, 0.1, lh.shape))

        # add unk tensor (more exactly, replace unk vectors)
        U = self.create_unk_for(mem)
        mem = mem + U

        # add noise
        D = mem.shape[-1]
        N = torch.normal(0, 1e-6 / D, mem.shape).to(mem.device)
        mem = mem + N

        tgt = self.create_right_shift_target(T)  # -> (S'+1, B, D)

        dec = self.decoder(mem, tgt)
        dec = dec[: tgt.shape[0] - 1]  # -> (S', B, D)
        dec = torch.transpose(dec, 0, 1)  # -> (B, S', D)
        self.context["decoded"] = dec
        h = dec

        W = self.bert.embeddings.word_embeddings.weight  # (V, D)
        h = torch.matmul(dec, W.T)  # (B, S', V)
        B, S, V = h.shape
        h = h.reshape(-1, V)  # -> (B*S, V)
        y = self.clf(h).reshape(B, S, V)

        return y

    def loss(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        loss = self._loss_end(y, t) + self._loss_middle()
        return loss

    def _loss_end(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = y.shape[0]
        _y = y.reshape((B, -1))  # -> (B, *)
        _t = t.reshape((B, -1))  # -> (B, *)
        loss = super().loss(y, t) + super().loss(_y, _t) + self._loss_difference(_y, _t)
        return loss

    def _loss_middle(self):
        dec: torch.Tensor = self.context["decoded"]  # -> (B, S', D)
        trg: torch.Tensor = self.context["target"]  # -> (B, S', D)
        B = dec.shape[0]
        dec = dec.reshape((B, -1))  # -> (B, *)
        trg = trg.reshape((B, -1))  # -> (B, *)
        loss = self._loss_difference(dec, trg)
        return loss

    def _loss_difference(self, y: torch.Tensor, t: torch.Tensor):
        assert y.shape == t.shape
        assert len(y.shape) > 1
        cos = self.cos(y, t)
        return self.mse(y, t) + self.mse(cos, torch.ones_like(cos))
