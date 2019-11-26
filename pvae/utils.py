import sys
import math
import time
import os
import shutil
import torch
import torch.distributions as dist
from torch.autograd import Variable, Function, grad


def lexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on left."""
    return A.expand(tuple(dimensions) + A.shape)


def rexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on right."""
    return A.view(A.shape + (1,)*len(dimensions)).expand(A.shape + tuple(dimensions))


def assert_no_nan(name, g):
    if torch.isnan(g).any(): raise Exception('nans in {}'.format(name))


def assert_no_grad_nan(name, x):
    if x.requires_grad: x.register_hook(lambda g: assert_no_nan(name, g))


# Classes
class Constants(object):
    eta = 1e-5
    log2 = math.log(2)
    logpi = math.log(math.pi)
    log2pi = math.log(2 * math.pi)
    logceilc = 88                # largest cuda v s.t. exp(v) < inf
    logfloorc = -104             # smallest cuda v s.t. exp(v) > 0
    invsqrt2pi = 1. / math.sqrt(2 * math.pi)
    sqrthalfpi = math.sqrt(math.pi/2)


def logsinh(x):
    # torch.log(sinh(x))
    return x + torch.log(1 - torch.exp(-2 * x)) - Constants.log2


def logcosh(x):
    # torch.log(cosh(x))
    return x + torch.log(1 + torch.exp(-2 * x)) - Constants.log2


class Arccosh(Function):
    # https://github.com/facebookresearch/poincare-embeddings/blob/master/model.py
    @staticmethod
    def forward(ctx, x):
        ctx.z = torch.sqrt(x * x - 1)
        return torch.log(x + ctx.z)

    @staticmethod
    def backward(ctx, g):
        z = torch.clamp(ctx.z, min=Constants.eta)
        z = g / z
        return z


class Arcsinh(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.z = torch.sqrt(x * x + 1)
        return torch.log(x + ctx.z)

    @staticmethod
    def backward(ctx, g):
        z = torch.clamp(ctx.z, min=Constants.eta)
        z = g / z
        return z


# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))


# Functions
def save_vars(vs, filepath):
    """
    Saves variables to the given filepath in a safe manner.
    """
    if os.path.exists(filepath):
        shutil.copyfile(filepath, '{}.old'.format(filepath))
    torch.save(vs, filepath)


def save_model(model, filepath):
    """
    To load a saved model, simply use
    `model.load_state_dict(torch.load('path-to-saved-model'))`.
    """
    save_vars(model.state_dict(), filepath)


def log_mean_exp(value, dim=0, keepdim=False):
    return log_sum_exp(value, dim, keepdim) - math.log(value.size(dim))


def log_sum_exp(value, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))


def log_sum_exp_signs(value, signs, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim))


def get_mean_param(params):
    """Return the parameter used to show reconstructions or generations.
    For example, the mean for Normal, or probs for Bernoulli.
    For Bernoulli, skip first parameter, as that's (scalar) temperature
    """
    if params[0].dim() == 0:
        return params[1]
    # elif len(params) == 3:
    #     return params[1]
    else:
        return params[0]


def probe_infnan(v, name, extras={}):
    nps = torch.isnan(v)
    s = nps.sum().item()
    if s > 0:
        print('>>> {} >>>'.format(name))
        print(name, s)
        print(v[nps])
        for k, val in extras.items():
            print(k, val, val.sum().item())
        quit()


def has_analytic_kl(type_p, type_q):
    return (type_p, type_q) in torch.distributions.kl._KL_REGISTRY
