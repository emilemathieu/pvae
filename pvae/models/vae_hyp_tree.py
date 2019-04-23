import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset

import math
import numpy as np
from sklearn.model_selection._split import _validate_shuffle_split
from .vae import VAE
from pvae.utils import Constants
from pvae.vis import array_plot

from pvae.distributions.riemannian_normal import RiemannianNormal
from pvae.distributions.wrapped_normal import WrappedNormal
from pvae.ops.poincare_layers import Hypergyroplane, MobiusLinear, MobiusNL
from pvae.ops.mobius_poincare import exp_map_zero, exp_map_zero_polar, log_map_zero, poincare_belongs
from pvae.datasets import SyntheticDataset

def extra_hidden_layer(hidden_dim, non_lin):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), non_lin)

# Classes
class Enc(nn.Module):
    """ Usual encoder followed by an exponential map """
    def __init__(self, latent_dim, data_dim, non_lin, c, num_hidden_layers=1, hidden_dim=100):
        super(Enc, self).__init__()
        self.c = c
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-1], -1))
        mu = self.fc21(e)          # flatten data
        mu = exp_map_zero(mu, self.c)
        return self.c, mu, F.softplus(.5*self.fc22(e)) + Constants.eta

class Dec(nn.Module):
    """ Usual encoder preceded by a logarithm map """
    def __init__(self, latent_dim, data_dim, non_lin, c, num_hidden_layers=1, hidden_dim=100):
        super(Dec, self).__init__()
        self.data_dim = data_dim
        self.c = c
        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        z = log_map_zero(z, self.c)
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], self.data_dim)  # reshape data
        return mu, torch.ones_like(mu)


class DecGyroplane(nn.Module):
    """ First layer is a Hypergyroplane followed by usual decoder """
    def __init__(self, latent_dim, data_dim, non_lin, c, num_hidden_layers=1, hidden_dim=100):
        super(DecGyroplane, self).__init__()
        self.data_dim = data_dim
        self.c = c
        modules = []
        modules.append(nn.Sequential(Hypergyroplane(latent_dim, hidden_dim, c=self.c), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], self.data_dim)  # reshape data
        return mu, torch.ones_like(mu)


class Tree(VAE):
    """ Derive a specific sub-class of a VAE for tree data. """
    def __init__(self, params):
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        super(Tree, self).__init__(
            eval(params.prior),           # prior distribution
            eval(params.posterior),       # posterior distribution
            dist.Normal,                  # likelihood distribution
            eval('Enc' + params.arch_enc)(params.latent_dim, *params.data_dim, getattr(nn, params.nl)(), c, params.num_hidden_layers, params.hidden_dim),
            eval('Dec' + params.arch_dec)(params.latent_dim, *params.data_dim, getattr(nn, params.nl)(), c, params.num_hidden_layers, params.hidden_dim),
            params
        )
        self.c = c
        self._pz_mu = nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)
        self.modelName = 'Tree'

    @property
    def pz_params(self):
        return self.c.mul(1), self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std_scale)

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        print('Load training data...')
        dataset = SyntheticDataset(*self.data_dim, *args)
        n_train, n_test = _validate_shuffle_split(len(dataset), test_size=None, train_size=0.7)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)
        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N, K = 10, 1
        _, _, samples = super(Tree, self).generate(N, K)
        array_plot([samples.data.cpu()], '{}/gen_samples_{:03d}.png'.format(runPath, epoch))

    def reconstruct(self, data, runPath, epoch):
        recon = super(Tree, self).reconstruct(data)
        array_plot([data.data.cpu(), recon.data.cpu()], '{}/reconstruct_{:03d}.png'.format(runPath, epoch))
