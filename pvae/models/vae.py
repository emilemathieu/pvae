# Base VAE class definition

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from pvae.utils import get_mean_param

class VAE(nn.Module):
    def __init__(self, prior_dist, posterior_dist, likelihood_dist, enc, dec, params):
        super(VAE, self).__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = posterior_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self.data_size = params.data_size
        self.prior_std = params.prior_std

        if self.px_z == dist.RelaxedBernoulli:
            self.px_z.log_prob = lambda self, value: \
                -F.binary_cross_entropy_with_logits(
                    self.probs if value.dim() <= self.probs.dim() else self.probs.expand_as(value),
                    value.expand(self.batch_shape) if value.dim() <= self.probs.dim() else value,
                    reduction='none'
                )

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        raise NotImplementedError

    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            mean_pz = get_mean_param(self.pz_params)
            mean = get_mean_param(self.dec(mean_pz))
            px_z_params = self.dec(self.pz(*self.pz_params).sample(torch.Size([N])))
            means = get_mean_param(px_z_params)
            samples = self.px_z(*px_z_params).sample(torch.Size([K]))

        return mean, \
            means.view(-1, *means.size()[2:]), \
            samples.view(-1, *samples.size()[3:])

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            px_z_params = self.dec(qz_x.rsample(torch.Size([1])).squeeze(0))

        return get_mean_param(px_z_params)

    def forward(self, x, K=1):
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std_scale)

    def init_last_layer_bias(self, dataset): pass
