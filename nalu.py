import torch
from torch import nn


class NAC(nn.Module):
    def __init__(self, in_dim, out_dim, init_fun=nn.init.xavier_uniform_):
        super().__init__()

        self._W_hat = nn.Parameter(torch.empty(in_dim, out_dim))
        self._M_hat = nn.Parameter(torch.empty(in_dim, out_dim))

        self.register_parameter('W_hat', self._W_hat)
        self.register_parameter('M_hat', self._M_hat)

        for param in self.parameters():
            init_fun(param)

    def forward(self, x):
        W = torch.tanh(self._W_hat) * torch.sigmoid(self._M_hat)
        return x.matmul(W)


class StackedNAC(nn.Module):
    def __init__(self, n_layers, in_dim, out_dim, hidden_dim,
                 init_fun=nn.init.xavier_uniform_):
        super().__init__()

        self._nac_stack = nn.Sequential(*[
            NAC(
                in_dim if i == 0 else hidden_dim,
                out_dim if i == n_layers - 1 else hidden_dim,
                init_fun=init_fun
            )
            for i in range(n_layers)
        ])

    def forward(self, x):
        return self._nac_stack(x)


class NALU(nn.Module):
    def __init__(self, in_dim, out_dim, init_fun=nn.init.xavier_uniform_):
        super().__init__()

        self._G = nn.Linear(in_dim, 1, bias=False)
        self._nac = NAC(in_dim, out_dim, init_fun=init_fun)
        self._epsilon = 1e-8
       
    def forward(self, x):
        g = torch.sigmoid(self._G(x))

        m = torch.exp(
            self._nac(torch.log(torch.abs(x) + self._epsilon))
        )
        a = self._nac(x)

        y = g * a + (1 - g) * m

        return y


class SNALU(nn.Module):
    def __init__(self, in_dim, out_dim, init_fun=nn.init.xavier_uniform_):
        super().__init__()

        self._G = nn.Linear(in_dim, 1)
        self._nac = NAC(in_dim, out_dim, init_fun=init_fun)
        self._epsilon = 1e-8

    def log_rescale(self, x):
        return torch.sign(x).prod(1).unsqueeze(1)*torch.log(torch.abs(x) + 1.0 + self._epsilon)
        
    def forward(self, x):
        g = torch.sigmoid(self._G(self.log_rescale(x)))

        m = torch.exp(
            self._nac(torch.log(torch.abs(x) + self._epsilon))
        )
        sm = torch.sign(x).prod(1).unsqueeze(1) * m
        a = self._nac(x)

        y = g * a + (1 - g) * sm

        return y
    

class StackedNALU(nn.Module):
    def __init__(self, n_layers, in_dim, out_dim, hidden_dim,
                 init_fun=nn.init.xavier_uniform_):
        super().__init__()

        self._nalu_stack = nn.Sequential(*[
            NALU(
                in_dim if i == 0 else hidden_dim,
                out_dim if i == n_layers - 1 else hidden_dim,
                init_fun=init_fun
            )
            for i in range(n_layers)
        ])

    def forward(self, x):
        return self._nalu_stack(x)

    
class StackedSNALU(nn.Module):
    def __init__(self, n_layers, in_dim, out_dim, hidden_dim,
                 init_fun=nn.init.xavier_uniform_):
        super().__init__()

        self._snalu_stack = nn.Sequential(*[
            SNALU(
                in_dim if i == 0 else hidden_dim,
                out_dim if i == n_layers - 1 else hidden_dim,
                init_fun=init_fun
            )
            for i in range(n_layers)
        ])

    def forward(self, x):
        return self._snalu_stack(x)
