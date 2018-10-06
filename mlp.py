import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_layers, in_dim, out_dim, hidden_dim, act,
                 init_fun=nn.init.xavier_uniform_):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(
                in_dim if i == 0 else hidden_dim,
                out_dim if i == n_layers - 1 else hidden_dim
            ))

            if i < n_layers - 1 and act is not None:
                layers.append(act())

        self._seq = nn.Sequential(*layers)

    def forward(self, x):
        return self._seq(x)
