import math
import torch
import torch.distributions as dist
from torch.distributions import constraints
from numbers import Number
from pvae.utils import Constants, logsinh, logcosh
from pvae.ops.mobius_poincare import exp_map_x_polar, poinc_dist_sq, poincare_belongs
from pvae.distributions.hyperbolic_radius import HyperbolicRadius
from pvae.distributions.hyperspherical_uniform import HypersphericalUniform

class RiemannianNormal(dist.Distribution):
    arg_constraints = {'loc': dist.constraints.interval(-1, 1), 'scale': dist.constraints.positive}
    support = dist.constraints.interval(-1, 1)
    has_rsample = True

    @property
    def mean(self):
        return self.loc
    
    def __init__(self, c, loc, scale, validate_args=None):
        assert not (torch.isnan(loc).any() or torch.isnan(scale).any())
        assert poincare_belongs(loc, c)
        self.c = c
        self.loc = loc
        self.scale = scale
        self.scale = torch.clamp(scale, min=.1)
        self.dim = self.loc.size(-1)
        self.radius = HyperbolicRadius(self.dim, self.c, self.scale)
        self.direction = HypersphericalUniform(self.dim - 1, device=loc.device)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(RiemannianNormal, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        alpha = self.direction.sample(torch.Size([*shape[:-1]]))
        radius = self.radius.rsample(sample_shape)
        res = exp_map_x_polar(self.loc.expand(shape), radius, alpha, self.c)
        return res

    def log_prob(self, value):
        loc = self.loc.expand(value.shape)
        radius_sq = poinc_dist_sq(loc, value, self.c)
        res = - radius_sq / 2 / self.scale.pow(2) - self.direction._log_normalizer() - self.radius.log_normalizer
        return res
