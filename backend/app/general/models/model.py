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
        add_noise=True,
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
        self.add_noise = add_noise

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
        tgt = torch.zeros(shp, device=T.device)
        tgt[1:] = T[:-1]
        return tgt

    def forward(self, *args, **kwargs):
        o = self.bert(*args, **kwargs)
        mem = torch.transpose(o["last_hidden_state"], 0, 1)  # -> (S, B, D)
        # po = o["pooler_output"]

        targets = self.context["targets"]
        to = self.bert(**targets)
        trg = to["last_hidden_state"]
        tgt = torch.transpose(trg, 0, 1)  # -> (S', B, D)
        self.context["trg"] = trg  # -> (B, S', D)

        # # NOTE: mem に揺らぎを与える
        # #   - onehot を UNKnown に変える / self.tokenizer.unk_token_id
        # #   - decoder の入力(bert の出力ベクトル)にノイズ (torch.normal(0, 0.1, lh.shape))
        if self.add_noise:
            # add unk tensor (more exactly, replace unk vectors)
            U = self.create_unk_for(mem)
            mem = mem + U

            # add noise
            D = mem.shape[-1]
            N = torch.normal(0, 1e-6 / D, mem.shape).to(mem.device)
            mem = mem + N

        W = self.bert.embeddings.word_embeddings.weight  # (V, D)

        tgt = self.create_right_shift_target(tgt)  # -> (S'+1, B, D)
        dec = self.decoder(tgt, mem)
        dec = torch.transpose(dec, 0, 1)  # -> (B, S', D)
        h = torch.matmul(dec, W.T)  # (B, S'+1, V)
        B, S, V = h.shape
        h = h.reshape(-1, V)  # -> (B*S, V)
        y = self.clf(h).reshape(B, S, V)

        # dec = dec[: tgt.shape[0] - 1]  # -> (S', B, D)
        # dec = self.decoder(tgt, mem)
        # dec = torch.transpose(dec, 0, 1)  # -> (B, S', D)
        # self.context["dec"] = dec
        # B, S, D = dec.shape
        # y = self.clf(dec).reshape(B, S, -1)
        return y

    def to_text(self, y_rec: torch.Tensor, do_argmax=True) -> torch.Tensor:
        tokenizer = self.context["tokenizer"]
        y = y_rec.argmax(dim=-1) if do_argmax else y_rec
        return "".join(tokenizer.decode(y))

    def loss(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # loss = self._loss_end(y, t) + self._loss_middle()
        # loss = self._loss_end(y, t)
        loss = super().loss(y, t)
        return loss

    def _loss_end(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = y.shape[0]
        _y = y.reshape((B, -1))  # -> (B, *)
        _t = t.reshape((B, -1))  # -> (B, *)
        loss_token = super().loss(y, t) + self._loss_difference(y, t)
        loss_seq = super().loss(_y, _t) + self._loss_difference(_y, _t)
        loss = loss_token + loss_seq
        return loss

    def _loss_middle(self):
        dec: torch.Tensor = self.context["dec"]  # -> (B, S', D)
        trg: torch.Tensor = self.context["trg"]  # -> (B, S', D)
        B = dec.shape[0]
        _dec = dec.reshape((B, -1))  # -> (B, *)
        _trg = trg.reshape((B, -1))  # -> (B, *)
        loss = self._loss_difference(dec, trg) + self._loss_difference(_dec, _trg)
        return loss

    def _loss_difference(self, y: torch.Tensor, t: torch.Tensor):
        assert y.shape == t.shape
        assert len(y.shape) > 1
        cos = self.cos(y, t)
        return self.mse(y, t) + self.mse(cos, torch.ones_like(cos))

    def calculate_scores(self, y: torch.Tensor, t: torch.Tensor) -> dict:
        import numpy
        from torchmetrics.functional import accuracy, bleu_score, rouge_score

        acc = accuracy(y, t, task="multiclass", num_classes=self.n_out)
        acc = acc.item()

        bleus = []
        rouges = {}
        for _y, _t in zip(y, t):
            pred_text = self.to_text(_y)
            labl_text = self.to_text(_t)
            b = bleu_score(pred_text, [labl_text])
            bleus.append(b)
            r = rouge_score(pred_text, labl_text)
            for k, v in r.items():
                v = v.item()
                if k in rouges:
                    rouges[k].append(v)
                else:
                    rouges[k] = [v]

        bleu = numpy.mean(bleus)
        scores = dict(accuracy=acc, bleu=bleu)

        for k, rouge_list in rouges.items():
            rouge = numpy.mean(rouge_list)
            scores[k] = rouge

        return scores
