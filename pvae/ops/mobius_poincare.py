# Code ported to PyTorch from https://github.com/dalab/hyperbolic_nn/blob/master/util.py

import torch
import math
from torch.autograd import Function, grad
from pvae.utils import Constants, logsinh, logcosh, Arccosh
import numpy as np

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0

def poincare_belongs(value, c):
    if len(value.shape) == 1: value = value.unsqueeze(0)
    return (c * value.pow(2).sum(-1) <= 1).all()

def project_hyp_vecs(x, c):
	# https://www.tensorflow.org/api_docs/python/tf/clip_by_norm
    # Projection op. Need to make sure hyperbolic embeddings are inside the unit ball.
    norm_x = norm(x)
    clip_norm = (1. - PROJ_EPS) / c.sqrt()
    intermediate = x * clip_norm
    return intermediate / torch.max(norm_x, clip_norm)


# ######################## x,y have shape [batch_size, emb_dim] in all functions ################

def atanh(x):
	x = torch.clamp(x, max=1. - EPS)
	return .5 * (torch.log(1 + x) - torch.log(1 - x))

def tanh(x):
   return torch.tanh(torch.clamp(torch.clamp(x, min=-MAX_TANH_ARG), max=MAX_TANH_ARG))

def dot(x, y):
    return (x * y).sum(-1, keepdim=True)

def norm(x):
	return x.pow(2).sum(-1, keepdim=True).sqrt()
    # return torch.norm(x, dim=-1, keepdim=True)


# #########################
def mob_add(u, v, c):
    v = v + EPS
    dot_u_v = 2. * c * dot(u, v)
    norm_u_sq = c * dot(u,u)
    norm_v_sq = c * dot(v,v)
    denominator = 1. + dot_u_v + norm_v_sq * norm_u_sq
    result = (1. + dot_u_v + norm_v_sq) / denominator * u + (1. - norm_u_sq) / denominator * v
    return project_hyp_vecs(result, c)


# #########################
def poinc_dist(u, v, c):
    m = mob_add(-u, v, c) + EPS
    atanh_x = c.sqrt() * norm(m)
    dist_poincare = 2. / c.sqrt() * atanh(atanh_x)
    return dist_poincare

def poinc_dist_sq(u, v, c):
    return poinc_dist(u, v, c) ** 2

def euclid_dist_sq(u, v):
    return (u - v).pow(2).sum(-1, keepdim=True)


# #########################
def mob_scalar_mul(r, v, c):
    v = v + EPS
    norm_v = norm(v)
    nomin = tanh(r * atanh(c.sqrt() * norm_v))
    result = nomin / (c.sqrt() * norm_v) * v
    return project_hyp_vecs(result, c)


# #########################
def lambda_x(x, c):
    return 2. / (1 - c * dot(x,x))

def parallel_transport_x(x, v, c):
    return (1 - c * dot(x,x)) * v

def parallel_transport_0(x, v, c):
    return v / (1 - c * dot(x,x))

def parallel_transport_y_x(y, x, v, c):
    return parallel_transport_x(x, parallel_transport_0(y, v, c), c)

def exp_map_zero_polar(r, v, c):
    v = v + EPS # Perturbe v to avoid dealing with v = 0
    norm_v = norm(v)
    result = (tanh(c.sqrt() * r / 2) / (c.sqrt() * norm_v)) * v
    return project_hyp_vecs(result, c)

def exp_map_x_polar(x, r, v, c):
    v = v + EPS # Perturbe v to avoid dealing with v = 0
    norm_v = norm(v)
    second_term = (tanh(c.sqrt() * r / 2) / (c.sqrt() * norm_v)) * v
    return mob_add(x, second_term, c)

def exp_map_x(x, v, c):
    # v = v + EPS # Perturbe v to avoid dealing with v = 0
    norm_v = norm(v)
    second_term = (tanh(c.sqrt() * lambda_x(x, c) * norm_v / 2) / (c.sqrt() * norm_v)) * v
    return mob_add(x, second_term, c)

def log_map_x(x, y, c):
    diff = mob_add(-x, y, c) + EPS
    norm_diff = norm(diff)
    lam = lambda_x(x, c)
    return ( ( (2. / c.sqrt()) / lam) * atanh(c.sqrt() * norm_diff) / norm_diff) * diff

def exp_map_zero(v, c):
    v = v + EPS # Perturbe v to avoid dealing with v = 0
    norm_v = norm(v)
    result = tanh(c.sqrt() * norm_v) / (c.sqrt() * norm_v) * v
    return project_hyp_vecs(result, c)

def log_map_zero(y, c):
    diff = y + EPS
    norm_diff = norm(diff)
    return 1. / c.sqrt() * atanh(c.sqrt() * norm_diff) / norm_diff * diff


# #########################
def mob_mat_mul(M, x, c, c_out=None):
    if c_out is None: c_out = c
    x = x + EPS
    Mx = torch.matmul(x, M) + EPS
    MX_norm = norm(Mx)
    x_norm = norm(x)
    result = 1. / c_out.sqrt() * tanh(MX_norm / x_norm * atanh(c.sqrt() * x_norm)) / MX_norm * Mx
    return project_hyp_vecs(result, c_out)


# # x is hyperbolic, u is Euclidean. Computes diag(u) \otimes x.
def mob_pointwise_prod(x, u, c):
    x = x + EPS
    Mx = x * u + EPS
    MX_norm = norm(Mx)
    x_norm = norm(x)
    result = 1. / c.sqrt() * tanh(MX_norm / x_norm * atanh(c.sqrt() * x_norm)) / MX_norm * Mx
    return project_hyp_vecs(result, c)
