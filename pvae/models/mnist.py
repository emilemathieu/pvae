import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms

import math
from numpy import prod
from .vae import VAE
from pvae.utils import Constants

from pvae.distributions import RiemannianNormal, WrappedNormal
from torch.distributions import Normal

from pvae import manifolds
from .architectures import EncLinear, DecLinear, EncWrapped, DecWrapped, EncMob, DecMob, DecGeo, DecBernouilliWrapper

data_size = torch.Size([1, 28, 28])


class Mnist(VAE):
    def __init__(self, params):
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        manifold = getattr(manifolds, params.manifold)(params.latent_dim, c)
        super(Mnist, self).__init__(
            eval(params.prior),   # prior distribution
            eval(params.posterior),   # posterior distribution
            dist.RelaxedBernoulli,        # likelihood distribution
            eval('Enc' + params.enc)(manifold, data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim, params.prior_iso),
            DecBernouilliWrapper(eval('Dec' + params.dec)(manifold, data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim)),
            params
        )
        self.manifold = manifold
        self.c = c
        self._pz_mu = nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)
        self.modelName = 'Mnist'

    def init_last_layer_bias(self, train_loader):
        if not hasattr(self.dec.dec.fc31, 'bias'): return
        with torch.no_grad():
            p = torch.zeros(prod(data_size[1:]), device=self._pz_mu.device)
            N = 0
            for i, (data, _) in enumerate(train_loader):
                data = data.to(self._pz_mu.device)
                B = data.size(0)
                N += B
                p += data.view(-1, prod(data_size[1:])).sum(0)
            p /= N
            p += 1e-4
            self.dec.dec.fc31.bias.set_(p.log() - (1 - p).log())

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std), self.manifold

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        # this is required if using the relaxedBernoulli because it doesn't
        # handle scoring values that are actually 0. or 1.
        tx = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda p: p.clamp(Constants.eta, 1 - Constants.eta))
        ])
        train_loader = DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(
            datasets.MNIST('data', train=False, download=True, transform=tx),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N, K = 64, 9
        mean, means, samples = super(Mnist, self).generate(N, K)
        save_image(mean.data.cpu(), '{}/gen_mean_{:03d}.png'.format(runPath, epoch))
        save_image(means.data.cpu(), '{}/gen_means_{:03d}.png'.format(runPath, epoch))

    def reconstruct(self, data, runPath, epoch):
        recon = super(Mnist, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon])
        save_image(comp.data.cpu(), '{}/recon_{:03d}.png'.format(runPath, epoch))
