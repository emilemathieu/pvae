import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms

import numpy as np
import math
from numpy import prod, sqrt
from .vae import VAE
from pvae.utils import Constants

data_size = torch.Size([1, 28, 28])
data_dim = int(prod(data_size))


def extra_hidden_layer(hidden_dim, non_lin):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), non_lin)

# Classes
class Enc(nn.Module):
    """ Generate latent parameters for MNIST. """
    def __init__(self, latent_dim, non_lin, prior_aniso, num_hidden_layers=1, hidden_dim=100):
        super(Enc, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim if prior_aniso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-3], -1)) # flatten data
        mu = self.fc21(e)
        return mu, F.softplus(self.fc22(e)).expand(mu.size()) + Constants.eta

class Dec(nn.Module):
    """ Generate observation parameters for MNIST. """
    def __init__(self, latent_dim, non_lin, num_hidden_layers=1, hidden_dim=100):
        super(Dec, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        p = self.fc31(self.dec(z))
        d = p.view(*z.size()[:-1], *data_size) # reshape data
        return torch.tensor(1.0).to(z.device), d


class Mnist(VAE):
    """ Derive a specific sub-class of a VAE for MNIST. """
    def __init__(self, params):
        super(Mnist, self).__init__(
            dist.Normal,           # prior distribution
            dist.Normal,           # posterior distribution
            dist.RelaxedBernoulli, # likelihood distribution
            Enc(params.latent_dim, getattr(nn, params.nl)(), params.prior_aniso, params.num_hidden_layers, params.hidden_dim),
            Dec(params.latent_dim, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim),
            params
        )
        self._pz_mu = nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)
        self.modelName = 'Mnist'

    def init_last_layer_bias(self, train_loader):
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
        save_image(mean.sigmoid().data.cpu(), '{}/gen_mean_{:03d}.png'.format(runPath, epoch))
        save_image(means.data.cpu(), '{}/gen_means_{:03d}.png'.format(runPath, epoch))

    def reconstruct(self, data, runPath, epoch):
        recon = super(Mnist, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon])
        save_image(comp.data.cpu(), '{}/recon_{:03d}.png'.format(runPath, epoch))
