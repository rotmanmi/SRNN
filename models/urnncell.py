import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
import torch.jit as jit

from typing import List, Tuple
from torch import Tensor


def normalize(x_real, x_imag):
    norm = torch.sqrt(comp_norm(x_real, x_imag))
    # norm = tf.sqrt(tf.reduce_sum(tf.abs(z)**2))
    factor = (norm + 1e-6)
    return x_real / factor, x_imag / factor


def magnitude(x_real, x_imag):
    return torch.sqrt(comp_sqr(x_real, x_imag))


# z: complex[batch_sz, num_units]
# bias: real[num_units]
def modReLU(z_real, z_imag, bias):  # relu(|z|+b) * (z / |z|)
    norm = magnitude(z_real, z_imag)

    scale = torch.relu(norm + bias) / (norm + 1e-6)
    # scaled = tf.complex(tf.real(z) * scale, tf.imag(z) * scale)
    return z_real * scale, z_imag * scale


def comp_mul(x_real, x_imag, y_real, y_imag):
    return x_real * y_real - x_imag * y_imag, x_real * y_imag + x_imag * y_real


def conj(x_real, x_imag):
    return x_real, -x_imag


def comp_sqr(x_real, x_imag):
    return (x_real ** 2 + x_imag ** 2)


def comp_norm(x_real, x_imag):
    return (x_real ** 2 + x_imag ** 2).sum(-1)


def comp_outer(x_real, x_imag, y_real, y_imag):
    return torch.ger(x_real, y_real) - torch.ger(x_imag, y_imag), torch.ger(x_real, y_imag) + torch.ger(x_imag, y_real)


def comp_matmul(w_real, w_imag, x_real, x_imag):
    x_real = x_real.expand(w_real.size(0), x_real.size(1), x_real.size(2))
    x_imag = x_imag.expand(w_imag.size(0), x_imag.size(1), x_imag.size(2))
    return (torch.bmm(w_real, x_real) - torch.bmm(w_imag, x_imag)).squeeze(1), (
            torch.bmm(w_real, x_imag) + torch.bmm(w_imag, x_real)).squeeze(1)


class DiagonalMatrix(jit.ScriptModule):
    def __init__(self, num_units):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(num_units).uniform_(-np.pi, np.pi))

    # [batch_sz, num_units]
    @jit.script_method
    def forward(self, z_real, z_imag):
        vec_real = torch.cos(self.w)
        vec_imag = torch.sin(self.w)
        # [num_units] * [batch_sz, num_units] -> [batch_sz, num_units]
        return comp_mul(vec_real, vec_imag, z_real, z_imag)


# Reflection unitary matrix
class ReflectionMatrix(jit.ScriptModule):
    def __init__(self, num_units):
        super().__init__()
        self.num_units = num_units

        self.v_re = nn.Parameter(torch.rand(num_units))
        self.v_im = nn.Parameter(torch.rand(num_units))
        init.uniform_(self.v_re, -1, 1)
        init.uniform_(self.v_im, -1, 1)

    # [batch_sz, num_units]
    @jit.script_method
    def forward(self, z_real, z_imag):
        """
        def mul(self, z):
                v = tf.expand_dims(self.v, 1) # [num_units, 1]
                vstar = tf.conj(v) # [num_units, 1]
                vstar_z = tf.matmul(z, vstar) #[batch_size, 1]
                sq_norm = tf.reduce_sum(tf.abs(self.v)**2) # [1]
                factor = (2 / tf.complex(sq_norm, 0.0))
                return z - factor * tf.matmul(vstar_z, tf.transpose(v))

        """

        # out_real, out_imag = comp_mul(z_real, z_imag, *conj(self.v_re, self.v_im))
        v_star = conj(self.v_re, self.v_im)
        vstar_z = comp_matmul(z_real.unsqueeze(1), z_imag.unsqueeze(1), v_star[0].unsqueeze(0).unsqueeze(-1),
                              v_star[1].unsqueeze(0).unsqueeze(-1))
        sq_norm = comp_norm(self.v_re, self.v_im)
        factor = 2.0 / sq_norm
        vvstar_z = comp_matmul(vstar_z[0].unsqueeze(1), vstar_z[1].unsqueeze(1), self.v_re.unsqueeze(0).unsqueeze(1),
                               self.v_im.unsqueeze(0).unsqueeze(1))
        return z_real - factor * vvstar_z[0], z_imag - factor * vvstar_z[1]


# Permutation unitary matrix

class PermutationMatrix(jit.ScriptModule):
    def __init__(self, num_units):
        super().__init__()
        self.num_units = num_units
        # perm = np.random.permutation(num_units)
        self.P = nn.Parameter(torch.randperm(num_units), requires_grad=False)
        # self.P = tf.constant(perm, tf.int32)

    # [batch_sz, num_units], permute columns
    @jit.script_method
    def forward(self, z_real, z_imag):
        return z_real[:, self.P], z_imag[:, self.P]
        # return tf.transpose(tf.gather(tf.transpose(z), self.P))


"""
def __init__(self, num_units, num_in, reuse=None):
        super(URNNCell, self).__init__(_reuse=reuse)
        # save class variables
        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units*2 
        self._output_size = num_units*2

        # set up input -> hidden connection
        self.w_ih = tf.get_variable("w_ih", shape=[2*num_units, num_in], 
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.b_h = tf.Variable(tf.zeros(num_units), # state size actually
                                    name="b_h")

        # elementary unitary matrices to get the big one
        self.D1 = DiagonalMatrix("D1", num_units)
        self.R1 = ReflectionMatrix("R1", num_units)
        self.D2 = DiagonalMatrix("D2", num_units)
        self.R2 = ReflectionMatrix("R2", num_units)
        self.D3 = DiagonalMatrix("D3", num_units)
        self.P = PermutationMatrix("P", num_units)
    # needed properties

"""


def FFT(x_real, x_imag):
    return [s.squeeze(-1) for s in torch.fft(torch.stack([x_real, x_imag], -1), 1, normalized=False).split(1, -1)]


def IFFT(x_real, x_imag):
    return [s.squeeze(-1) for s in torch.ifft(torch.stack([x_real, x_imag], -1), 1, normalized=False).split(1, -1)]


class URNNCell(jit.ScriptModule):
    __constants__ = ['_num_in', '_num_units', '_state_size', '_output_size']

    def __init__(self, num_units, num_in):
        super().__init__()
        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units * 2
        self._output_size = num_units * 2
        #
        self.b_h = nn.Parameter(torch.zeros(num_units))

        self.D1 = DiagonalMatrix(num_units)
        self.R1 = ReflectionMatrix(num_units)
        self.D2 = DiagonalMatrix(num_units)
        self.R2 = ReflectionMatrix(num_units)
        self.D3 = DiagonalMatrix(num_units)
        self.P = PermutationMatrix(num_units)

    @jit.script_method
    def forward(self, inputs, state):
        """The most basic URNN cell.
        Args:
            inputs (Tensor - batch_sz x num_in): One batch of cell input.
            state (Tensor - batch_sz x num_units): Previous cell state: COMPLEX
        Returns:
        A tuple (outputs, state):
            outputs (Tensor - batch_sz x num_units*2): Cell outputs on the whole batch.
            state (Tensor - batch_sz x num_units): New state of the cell.
        """
        # print("cell.call inputs:", inputs.shape, inputs.dtype)
        # print("cell.call state:", state.shape, state.dtype)

        # prepare input linear combination
        # inputs_mul = self.itoh(inputs)
        inputs_mul = inputs
        # inputs_mul = tf.matmul(inputs, tf.transpose(self.w_ih))  # [batch_sz, 2*num_units]
        inputs_c = [inputs_mul[:, :self._num_units], inputs_mul[:, self._num_units:]]
        # inputs_mul_c = tf.complex(inputs_mul[:, :self._num_units],
        #                           inputs_mul[:, self._num_units:])
        # [batch_sz, num_units]

        # prepare state linear combination (always complex!)
        state_c = [state[:, :self._num_units], state[:, self._num_units:]]
        state_mul = self.D1(state_c[0], state_c[1])
        state_mul = FFT(state_mul[0], state_mul[1])
        state_mul = self.R1(state_mul[0], state_mul[1])
        state_mul = self.P(state_mul[0], state_mul[1])
        state_mul = self.D2(state_mul[0], state_mul[1])
        state_mul = IFFT(state_mul[0], state_mul[1])
        state_mul = self.R2(state_mul[0], state_mul[1])
        state_mul = self.D3(state_mul[0], state_mul[1])
        # [batch_sz, num_units]

        # calculate preactivation
        preact = [inputs_c[0] + state_mul[0], inputs_c[1] + state_mul[1]]
        # [batch_sz, num_units]
        new_state_c = modReLU(preact[0], preact[1], self.b_h)  # [batch_sz, num_units] C
        new_state = torch.cat(new_state_c, -1)
        # new_state = tf.concat([tf.real(new_state_c), tf.imag(new_state_c)], 1)  # [batch_sz, 2*num_units] R
        # outside network (last dense layer) is ready for 2*num_units -> num_out
        output = new_state
        # print("cell.call output:", output.shape, output.dtype)
        # print("cell.call new_state:", new_state.shape, new_state.dtype)

        return output, new_state


class URNNsc(jit.ScriptModule):
    __constants__ = ['do_embedding']

    def __init__(self, input_size, output_size, hidden_size, *args, **kwargs):
        super().__init__()
        self.rnncell = URNNCell(hidden_size, input_size)
        self.end_fc = nn.Linear(hidden_size * 2, output_size)
        self.single_output = kwargs['single_output']
        self.itoh = nn.Linear(input_size, 2 * hidden_size, bias=False)
        self.hidden_size = hidden_size * 2
        init.xavier_uniform_(self.itoh.weight)
        self.do_embedding = kwargs['embedding'] or False
        self.embedding = nn.Embedding(input_size, 2 * hidden_size)

    @jit.script_method
    def forward(self, x, h_i):
        # batch_size, seq_len, inp_size = x.shape
        # outputs = torch.jit.annotate(List[Tensor], [])
        outputs = []
        if self.do_embedding:
            x = self.embedding(x.squeeze(-1))
        else:
            x = self.itoh(x)
        x = x.unbind(1)
        for time_step in range(len(x)):
            x_out, h_i = self.rnncell(x[time_step], h_i)
            outputs.append(x_out)

        outputs = torch.stack(outputs, 1)
        outputs = self.end_fc(outputs)
        return outputs, h_i


class URNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, *args, **kwargs):
        super().__init__()
        self.cell = URNNsc(input_size, output_size, hidden_size, *args, **kwargs)
        self.single_output = kwargs['single_output']

    def forward(self, x, h_i=None):
        factor = math.sqrt(1.5 / (self.cell.rnncell._num_units))
        if h_i is None:
            h_i = torch.zeros(x.shape[0], self.cell.rnncell._num_units * 2).uniform_(-factor, factor).to(x.device)
        outputs, hidden = self.cell(x, h_i)
        if self.single_output:
            outputs = outputs[:, -1, :].unsqueeze(1)

        return outputs, hidden


if __name__ == '__main__':
    s = ReflectionMatrix(4)
    p = PermutationMatrix(4)
    z_real, z_imag = torch.rand(2, 7, 4)
    s(z_real, z_imag)
    p(z_real, z_imag)

    u = URNNCell(7, 4)
    inp = torch.rand(16, 4)
    state = torch.rand(16, 14)
    u(inp, state)
