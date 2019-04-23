import torch
import torch.distributions as dist
from numpy import prod
from pvae.utils import has_analytic_kl, log_mean_exp
import torch.nn.functional as F

def vae_objective(model, x, K=1, beta=1.0, components=False, analytical_kl=False, **kwargs):
    """Computes E_{p(x)}[ELBO] """
    qz_x, px_z, zs = model(x, K)
    _, B, D = zs.size()
    flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
    lpx_z = px_z.log_prob(x.expand(px_z.batch_shape)).view(flat_rest).sum(-1)

    pz = model.pz(*model.pz_params)
    kld = dist.kl_divergence(qz_x, pz).unsqueeze(0).sum(-1) if \
        has_analytic_kl(type(qz_x), model.pz) and analytical_kl else \
        qz_x.log_prob(zs).sum(-1) - pz.log_prob(zs).sum(-1)

    obj = -lpx_z.mean(0).sum() + beta * kld.mean(0).sum()
    return (qz_x, px_z, lpx_z, kld, obj) if components else obj

def _iwae_objective_vec(model, x, K):
    """Helper for IWAE estimate for log p_\theta(x) -- full vectorisation."""
    qz_x, px_z, zs = model(x, K)
    flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x.expand(zs.size(0), *x.size())).view(flat_rest).sum(-1)
    lqz_x = qz_x.log_prob(zs).sum(-1)
    obj = lpz.squeeze(-1) + lpx_z.view(lpz.squeeze(-1).shape) - lqz_x.squeeze(-1)
    return -log_mean_exp(obj).sum()


def iwae_objective(model, x, K):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    Appropriate negation (for minimisation) happens in the helper
    """
    split_size = int(x.size(0) / (K * prod(x.size()) / (3e7)))  # rough heuristic
    if split_size >= x.size(0):
        obj = _iwae_objective_vec(model, x, K)
    else:
        obj = 0
        for bx in x.split(split_size):
            obj = obj + _iwae_objective_vec(model, bx, K)
    return obj
