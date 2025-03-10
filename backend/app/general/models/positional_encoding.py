import math

import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    # cf. https://nlp.seas.harvard.edu/2018/04/03/attention.html#decoder

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # pe: (max_S, D) / position: (max_S, 1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # -> (1, max_S, D)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, S, D)
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)
