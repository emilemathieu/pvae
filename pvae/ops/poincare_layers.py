import math

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
from pvae.ops.mobius_poincare import *
from pvae.utils import Arcsinh, Constants

class PoincareLayer(nn.Module):
    def __init__(self, in_features, out_features, c):
        super(PoincareLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c

    def get_bias(self):
        raise NotImplementedError

    def forward(self, input):
        raise NotImplementedError


class Hypergyroplane(PoincareLayer):
    def __init__(self, in_features, out_features, bias=True, c=torch.ones(1)):
        super(Hypergyroplane, self).__init__(
            in_features,
            out_features,
            c
        )
        self._weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self._bias = Parameter(torch.Tensor(out_features, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    @property
    def weight(self):
        return parallel_transport_x(self.get_bias(), self._weight, self.c) # weight \in T_0 => weight \in T_bias 

    def get_bias(self):
        return exp_map_zero(self._weight * self._bias, self.c) # reparameterisation of a point on the manifold

    def reset_parameters(self):
        init.kaiming_normal_(self._weight, a=math.sqrt(5))
        if self.get_bias() is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self._weight)
            bound = 4 / math.sqrt(fan_in)
            init.uniform_(self._bias, -bound, bound)

    def forward(self, input):
        shape = input.shape
        input = input.view(-1, self.in_features)
        B = input.shape[0]
        bias = self.get_bias()
        weight = self.weight
        norm_weight = weight.pow(2).sum(-1, keepdim=False).sqrt()
        # poincare_norm_weight = 2 * self._weight.pow(2).sum(-1) #lambda_x(bias, self.c).t() * norm_weight
        bias = bias.expand(B, self.out_features, self.in_features)
        input = input.unsqueeze(1).expand(B, self.out_features, self.in_features)
        dir_log_input = mob_add(-bias, input, self.c)
        denom = torch.clamp((1 - self.c * dir_log_input.pow(2).sum(-1)) * norm_weight, min=Constants.eta)
        hyperplane_dist = Arcsinh.apply(2 * self.c.sqrt() * (dir_log_input * weight).sum(-1) / denom) / self.c.sqrt()
        res = hyperplane_dist
        res = res.view(*shape[:-1], self.out_features)
        return res

class MobiusLinear(PoincareLayer):
    def __init__(self, in_features, out_features, bias=True, c=torch.ones(1), c_out=None):
        super(MobiusLinear, self).__init__(
            in_features,
            out_features,
            c
        )
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.c_out = c if (c_out is None) else c_out
        if bias:
            self._bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def get_bias(self):
        return exp_map_zero(self._bias, self.c_out) # reparameterisation of a point on the disk

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # usual init for weights
        if self.get_bias() is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self._bias, -bound, bound)

    def forward(self, input):
        y = mob_mat_mul(self.weight.t(), input.view(-1, self.in_features), self.c, self.c_out)
        if self.get_bias() is not None:
            y = mob_add(y, self.get_bias(), self.c_out)
        return y

class MobiusNL(nn.Module):
    def __init__(self, non_lin, hyp_output=True, c=torch.ones(1), c_out=None):
        super(MobiusNL, self).__init__()
        self.non_lin = non_lin
        self.hyp_output = hyp_output
        self.c = c
        self.c_out = c if (c_out is None) else c_out

    def forward(self, input):
        if self.non_lin is None:
            if self.hyp_output:
                return input
            else:
                return log_map_zero(input, self.c)

        eucl_h = self.non_lin(log_map_zero(input, self.c))

        if self.hyp_output:
            return exp_map_zero(eucl_h, self.c_out)
        else:
            return eucl_h

class ExpZero(nn.Module):
    def __init__(self, c):
        super(ExpZero, self).__init__()
        self.c = c

    def forward(self, input):
        return exp_map_zero(input, self.c)

class LogZero(nn.Module):
    def __init__(self, c):
        super(LogZero, self).__init__()
        self.c = c

    def forward(self, input):
        return log_map_zero(input, self.c)
