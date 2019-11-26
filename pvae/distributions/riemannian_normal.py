import torch
import torch.distributions as dist
from torch.distributions import constraints
from numbers import Number
from pvae.distributions.hyperbolic_radius import HyperbolicRadius
from pvae.distributions.hyperspherical_uniform import HypersphericalUniform


class RiemannianNormal(dist.Distribution):
    arg_constraints = {'loc': dist.constraints.interval(-1, 1), 'scale': dist.constraints.positive}
    support = dist.constraints.interval(-1, 1)
    has_rsample = True

    @property
    def mean(self):
        return self.loc
    
    def __init__(self, loc, scale, manifold, validate_args=None):
        assert not (torch.isnan(loc).any() or torch.isnan(scale).any())
        self.manifold = manifold
        self.loc = loc
        self.manifold.assert_check_point_on_manifold(self.loc)
        self.scale = scale.clamp(min=0.1, max=7.)
        self.radius = HyperbolicRadius(manifold.dim, manifold.c, self.scale)
        self.direction = HypersphericalUniform(manifold.dim - 1, device=loc.device)
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
        # u = radius * alpha / self.manifold.lambda_x(self.loc, keepdim=True)
        # res = self.manifold.expmap(self.loc, u)
        res = self.manifold.expmap_polar(self.loc, alpha, radius)
        return res

    def log_prob(self, value):
        loc = self.loc.expand(value.shape)
        radius_sq = self.manifold.dist(loc, value, keepdim=True).pow(2)
        res = - radius_sq / 2 / self.scale.pow(2) - self.direction._log_normalizer() - self.radius.log_normalizer
        return res
