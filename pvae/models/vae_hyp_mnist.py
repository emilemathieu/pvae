import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms

import math
import numpy as np
from numpy import prod, sqrt
from .vae import VAE
from pvae.utils import Constants

from pvae.distributions.riemannian_normal import RiemannianNormal
from pvae.distributions.wrapped_normal import WrappedNormal
from pvae.ops.poincare_layers import Hypergyroplane, MobiusLinear, MobiusNL, LogZero
from pvae.ops.mobius_poincare import exp_map_zero, exp_map_zero_polar, log_map_zero

data_size = torch.Size([1, 28, 28])
data_dim = int(prod(data_size))

def extra_hidden_layer(hidden_dim, non_lin):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), non_lin)

# Classes
class Enc(nn.Module):
    """ Usual encoder followed by an exponential map """
    def __init__(self, latent_dim, non_lin, c, num_hidden_layers=1, hidden_dim=100):
        super(Enc, self).__init__()
        self.c = c
        self.latent_dim = latent_dim
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-3], -1))            # flatten data
        mu = self.fc21(e)          # flatten data
        mu = exp_map_zero(mu, self.c)
        scale = F.softplus(self.fc22(e)) + Constants.eta
        return self.c, mu, scale

class Dec(nn.Module):
    """ First layer is a Hypergyroplane followed by usual decoder """
    def __init__(self, latent_dim, non_lin, c, num_hidden_layers=1, hidden_dim=100):
        super(Dec, self).__init__()
        self.c = c
        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        z = log_map_zero(z, self.c)
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *data_size)  # reshape data
        return torch.tensor(1.0).to(z.device), mu


class DecGyroplane(nn.Module):
    """ First layer is a Hypergyroplane followed by usual decoder """
    def __init__(self, latent_dim, non_lin, c, num_hidden_layers=1, hidden_dim=100):
        super(DecGyroplane, self).__init__()
        self.c = c
        modules = []
        modules.append(nn.Sequential(Hypergyroplane(latent_dim, hidden_dim, c=self.c), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *data_size)  # reshape data
        return torch.tensor(1.0).to(z.device), mu


class EncMob(nn.Module):
    """ Last layer is a Mobius layers """
    def __init__(self, latent_dim, non_lin, c, num_hidden_layers=1, hidden_dim=100):
        super(EncMob, self).__init__()
        self.c = c
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = MobiusLinear(hidden_dim, latent_dim, c=self.c)
        self.fc22 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-3], -1))            # flatten data
        mu = self.fc21(exp_map_zero(e, self.c))          # flatten data
        logvar = self.fc22(e)
        return self.c, mu, F.softplus(logvar) + Constants.eta

class DecMob(nn.Module):
    """ First layer is a Mobius Matrix multiplication """
    def __init__(self, latent_dim, non_lin, c, num_hidden_layers=1, hidden_dim=100):
        super(DecMob, self).__init__()
        self.c = c
        modules = []
        modules.append(nn.Sequential(MobiusLinear(latent_dim, hidden_dim, c=self.c), LogZero(self.c), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *data_size)  # reshape data
        return torch.tensor(1.0).to(z.device), mu


class EncMobFull(nn.Module):
    """ All layers are  Mobius layers """
    def __init__(self, latent_dim, non_lin, c, num_hidden_layers=1, hidden_dim=100):
        super(EncMobFull, self).__init__()
        self.c = c
        modules = []
        modules.append(nn.Sequential(MobiusLinear(data_dim, hidden_dim, c=self.c), MobiusNL(non_lin, c=self.c)))
        modules.extend([nn.Sequential(MobiusLinear(hidden_dim, hidden_dim, c=self.c), MobiusNL(non_lin, c=self.c)) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = MobiusLinear(hidden_dim, latent_dim, c=self.c)
        self.fc22 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = exp_map_zero(x, self.c)
        e = self.enc(x.view(*x.size()[:-3], -1))            # flatten data
        mu = self.fc21(e)          # flatten data
        logvar = self.fc22(log_map_zero(e, self.c))
        return self.c, mu, F.softplus(logvar) + Constants.eta

class DecMobFull(nn.Module):
    """ All layers are  Mobius layers """
    def __init__(self, latent_dim, non_lin, c, num_hidden_layers=1, hidden_dim=100):
        super(DecMobFull, self).__init__()
        self.c = c
        modules = []
        modules.append(nn.Sequential(MobiusLinear(latent_dim, hidden_dim, c=self.c), MobiusNL(non_lin, c=self.c)))
        modules.extend([nn.Sequential(MobiusLinear(hidden_dim, hidden_dim, c=self.c), MobiusNL(non_lin, c=self.c)) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = MobiusLinear(hidden_dim, data_dim, c=self.c)

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *data_size)  # reshape data
        mu = log_map_zero(mu, self.c)
        return torch.tensor(1.0).to(z.device), mu


class Mnist(VAE):
    def __init__(self, params):
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        super(Mnist, self).__init__(
            eval(params.prior),   # prior distribution
            eval(params.posterior),   # posterior distribution
            dist.RelaxedBernoulli,        # likelihood distribution
            eval('Enc' + params.arch_enc)(params.latent_dim, getattr(nn, params.nl)(), c, params.num_hidden_layers, params.hidden_dim),
            eval('Dec' + params.arch_dec)(params.latent_dim, getattr(nn, params.nl)(), c, params.num_hidden_layers, params.hidden_dim),
            params
        )
        self.c = c
        self._pz_mu = nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)
        self.modelName = 'Mnist'

    def init_last_layer_bias(self, train_loader):
        if not hasattr(self.dec.fc31, 'bias'): return
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
            self.dec.fc31.bias.set_(p.log() - (1 - p).log())

    @property
    def pz_params(self):
        return self.c.mul(1), self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std_scale)

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
