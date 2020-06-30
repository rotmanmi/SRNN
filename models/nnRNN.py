import torch
import torch.nn as nn
import torch.jit as jit
import numpy as np
from .exp_numpy import expm, expm_frechet
from .initializations import cayley_init, henaff_init, random_orthogonal_init


class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class OrthoRNNCell(nn.Module):
    def __init__(self, inp_size, hid_size, nonlin, bias=False, r_initializer=cayley_init,
                 i_initializer=nn.init.xavier_normal_):
        super(OrthoRNNCell, self).__init__()
        self.hidden_size = hid_size
        # Add Non linearity
        if nonlin == 'relu':
            self.nonlinearity = nn.ReLU()
        if nonlin == 'modrelu':
            self.nonlinearity = modrelu(hid_size)
        elif nonlin == 'tanh':
            self.nonlinearity = nn.Tanh()
        elif nonlin == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        else:
            self.nonlinearity = None

        # Create linear layer to act on input X
        self.U = nn.Linear(inp_size, hid_size, bias=bias)

        self.i_initializer = i_initializer
        self.r_initializer = r_initializer

        # determine if P is learnable, if P is learnable determine how

        self.log_P = torch.Tensor(hid_size, hid_size)
        self.log_P = nn.Parameter(self.log_P)

        self.P = torch.Tensor(hid_size, hid_size)
        self.P = nn.Parameter(self.P)

        UppT = torch.zeros(hid_size, hid_size)
        self.UppT = UppT
        self.UppT = nn.Parameter(self.UppT)
        self.M = torch.triu(torch.ones_like(self.UppT), diagonal=1)
        self.D = torch.zeros_like(self.UppT)

        # Create rotations and mask M for *.3 and *.4
        self.thetas = [0] * int(hid_size)
        for i in range(0, len(self.thetas)):
            self.thetas[i] = nn.Parameter(torch.Tensor([np.random.uniform(0, 2 * 3.14)]))
            self.register_parameter('theta_{}'.format(i), self.thetas[i])

        self.alpha_crit = nn.MSELoss()
        self.alphas = [0] * int(hid_size / 2)
        for i in range(0, len(self.alphas)):
            self.alphas[i] = nn.Parameter(torch.Tensor([np.random.uniform(1.00, 1.00)]))
            self.register_parameter('alpha_{}'.format(i), self.alphas[i])

        self.reset_parameters()

        # cudafy variables if needed
        # if cuda:
        self.P.data = self.P.data.cuda()
        self.log_P.data = self.log_P.data.cuda()
        self.M = self.M.cuda()
        self.D = self.D.cuda()
        for item in self.thetas:
            item = item.cuda()
        for item in self.alphas:
            item = item.cuda()

    def reset_parameters(self):
        if self.r_initializer == random_orthogonal_init or \
                self.r_initializer == henaff_init or \
                self.r_initializer == cayley_init:
            self.P.data = self._B(
                torch.as_tensor(self.r_initializer(self.hidden_size), dtype=torch.float32))
        else:
            self.r_initializer(self.P.data)

    def _A(self, gradients=False):
        A = self.log_P
        if not gradients:
            A = A.data
        A = A.triu(diagonal=1)
        return A - A.t()

    def _B(self, gradients=False):
        return expm(self._A())

    def orthogonal_step(self, optimizer):
        A = self._A(False)
        B = self.P.data
        if self.P.grad is None:
            self.P.grad = torch.zeros_like(B)
        G = self.P.grad.data
        BtG = B.t().mm(G)
        grad = 0.5 * (BtG - BtG.t())
        frechet_deriv = B.mm(expm_frechet(-A, grad))
        self.log_P.grad = (frechet_deriv - frechet_deriv.t()).triu(diagonal=1)
        optimizer.step()
        self.P.data = self._B(False)
        self.P.grad.data.zero_()

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = x.new_zeros(x.shape[0], self.hidden_size, requires_grad=True)
            self.first_hidden = hidden
            self.calc_rec()
        h = self.U(x) + torch.matmul(hidden, self.rec)
        if self.nonlinearity:
            h = self.nonlinearity(h)

        return h

    def calc_rec(self):
        self.calc_D()
        self.calc_alpha_block()
        self.rec = torch.matmul(
            torch.matmul(self.P, torch.mul(self.UppT, self.M) + torch.mul(self.alpha_block, self.D)), self.P.t())

    def calc_D(self):
        self.D = torch.zeros_like(self.UppT)
        for i in range(0, self.hidden_size, 2):
            self.D[i, i] = self.thetas[int(i / 2)].cos()
            self.D[i, i + 1] = -self.thetas[int(i / 2)].sin()
            self.D[i + 1, i] = self.thetas[int(i / 2)].sin()
            self.D[i + 1, i + 1] = self.thetas[int(i / 2)].cos()

    def calc_alpha_block(self):
        self.alpha_block = torch.zeros_like(self.UppT)
        for i in range(0, self.hidden_size, 2):
            self.alpha_block[i, i] = self.alphas[int(i / 2)]
            self.alpha_block[i + 1, i] = self.alphas[int(i / 2)]
            self.alpha_block[i, i + 1] = self.alphas[int(i / 2)]
            self.alpha_block[i + 1, i + 1] = self.alphas[int(i / 2)]

    def alpha_loss(self, lam):
        reg_loss = 0
        for alph in range(len(self.alphas)):
            reg_loss += lam * self.alpha_crit(self.alphas[alph], torch.ones_like(self.alphas[alph]))
        return reg_loss
