import torch
import torch.nn as nn
import torch.jit as jit
from typing import Optional, Tuple, List


def roll(tensor, size: int = 1, cut_half: bool = False):
    t_size = tensor.shape[-1]
    if cut_half:
        return torch.cat([tensor[..., :t_size // 2], torch.roll(tensor[..., t_size // 2:], size, -1)], -1)
    else:
        return torch.roll(tensor, size, -1)


class SGRUCell(jit.ScriptModule):
    """
        .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(hidden_size, hidden_size)
        self.Wz = nn.Linear(hidden_size, hidden_size)
        self.Wi = nn.Linear(hidden_size, hidden_size)
        self.bias_r = nn.Parameter(torch.randn(hidden_size))
        self.bias_z = nn.Parameter(torch.randn(hidden_size))
        self.bias_n = nn.Parameter(torch.randn(hidden_size))

        self.Sr = nn.Linear(hidden_size, hidden_size)
        self.Sz = nn.Linear(hidden_size, hidden_size)
        self.Si = nn.Linear(hidden_size, hidden_size)

    @jit.script_method
    def forward(self, input, state: Optional[torch.Tensor] = None):
        if state is None:
            r_t = torch.sigmoid(self.Wr(input))
            z_t = torch.sigmoid(self.Wz(input))
            n_t = torch.tanh((input))
            h_t = (1 - z_t) * n_t

        elif True:
            rolled_state = roll(state, 1)
            r_t = torch.sigmoid(self.Wr(input) + (rolled_state))
            z_t = torch.sigmoid(self.Wz(input) + (rolled_state))
            n_t = torch.tanh((input) + r_t * (rolled_state))
            h_t = (1 - z_t) * n_t + z_t * state

        else:
            rolled_state = roll(state, 1)
            r_t = torch.sigmoid(self.Wr(input) + rolled_state)
            z_t = torch.sigmoid(self.Wz(input) + rolled_state)
            n_t = torch.tanh((input) + r_t * rolled_state)
            h_t = (1 - z_t) * n_t + z_t * rolled_state

        return h_t, h_t


class GRUCell(jit.ScriptModule):
    """
        .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(hidden_size, hidden_size)
        self.Wz = nn.Linear(hidden_size, hidden_size)
        self.Wi = nn.Linear(hidden_size, hidden_size)
        self.bias_r = nn.Parameter(torch.randn(hidden_size))
        self.bias_z = nn.Parameter(torch.randn(hidden_size))
        self.bias_n = nn.Parameter(torch.randn(hidden_size))

        self.Sr = nn.Linear(hidden_size, hidden_size)
        self.Sz = nn.Linear(hidden_size, hidden_size)
        self.Si = nn.Linear(hidden_size, hidden_size)

    @jit.script_method
    def forward(self, input, state: Optional[torch.Tensor] = None):
        if state is None:
            r_t = torch.sigmoid(self.Wr(input))
            z_t = torch.sigmoid(self.Wz(input))
            n_t = torch.tanh(self.Wi(input))
            h_t = (1 - z_t) * n_t

        else:
            r_t = torch.sigmoid(self.Wr(input) + self.Sr(state))
            z_t = torch.sigmoid(self.Wz(input) + self.Sz(state))
            n_t = torch.tanh(self.Wi(input) + r_t * self.Si(state))
            h_t = (1 - z_t) * n_t + z_t * state

        return h_t, h_t


class SGRULayer(jit.ScriptModule):
    def __init__(self, *cell_args):
        super().__init__()
        self.cell = SGRUCell(*cell_args)

    @jit.script_method
    def forward(self, input, state: Optional[torch.Tensor] = None):
        inputs = input.unbind(1)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs, 1), state


class GRULayer(jit.ScriptModule):
    def __init__(self, *cell_args):
        super().__init__()
        self.cell = GRUCell(*cell_args)

    @jit.script_method
    def forward(self, input, state: Optional[torch.Tensor] = None):
        inputs = input.unbind(1)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs, 1), state
