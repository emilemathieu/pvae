import math
import torch
from torch.distributions.utils import _standard_normal

class HypersphericalUniform(torch.distributions.Distribution):
    """ source: https://github.com/nicola-decao/s-vae-pytorch/blob/master/hyperspherical_vae/distributions/von_mises_fisher.py """

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim
    
    def __init__(self, dim, device='cpu', validate_args=None):
        super(HypersphericalUniform, self).__init__(torch.Size([dim]), validate_args=validate_args)
        self._dim = dim
        self._device = device

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = torch.Size([*sample_shape, self._dim + 1])
        output = _standard_normal(shape, dtype=torch.float, device=self._device)

        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self):
        return self.__log_surface_area()
    
    def log_prob(self, x):
        return - torch.ones(x.shape[:-1]).to(self._device) * self._log_normalizer()

    def _log_normalizer(self):
        return self._log_surface_area().to(self._device)

    def _log_surface_area(self):
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - torch.lgamma(
            torch.Tensor([(self._dim + 1) / 2]))
