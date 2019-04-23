import math
import torch
import torch.distributions as dist
from torch.distributions import constraints
from numbers import Number
from pvae.utils import Constants, logsinh, logcosh
from torch.distributions.utils import _standard_normal, broadcast_all
from pvae.ops.mobius_poincare import exp_map_x_polar, exp_map_x, poinc_dist, poinc_dist_sq, lambda_x, poincare_belongs

def log_x_div_sinh(x, c):
    """ Stable function for torch.sinh(c.sqrt() * x).log() """
    res = c.sqrt().log() + x.log() - logsinh(c.sqrt() * x)
    zero_value_idx = x == 0.
    res[zero_value_idx] = 0.
    return res

class WrappedNormal(dist.Distribution):
    arg_constraints = {'loc': dist.constraints.interval(-1, 1), 'scale': dist.constraints.positive}
    support = dist.constraints.interval(-1, 1)
    has_rsample = True

    @property
    def mean(self):
        return self.loc
    
    def __init__(self, c, loc, scale, validate_args=None):
        assert not (torch.isnan(loc).any() or torch.isnan(scale).any())
        assert poincare_belongs(loc, c)
        self.loc = loc
        self.scale = torch.clamp(scale, min=.01)
        self.c = c
        self.dim = self.loc.size(-1)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(WrappedNormal, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        v = self.scale * _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        r = v.norm(dim=-1, keepdim=True)
        res = exp_map_x_polar(self.loc.expand(shape), r, v, self.c)
        return res

    def log_prob(self, value):
        loc = self.loc.expand(value.shape)
        radius = poinc_dist(loc, value, self.c)
        radius_sq = radius.pow(2)
        res = - radius_sq / 2 / self.scale.pow(2) \
            + (self.dim -1) * log_x_div_sinh(radius, self.c) \
            - self.dim * (math.log(math.sqrt(2 * math.pi)) + self.scale.log())

        assert not torch.isnan(res).any()
        return res
