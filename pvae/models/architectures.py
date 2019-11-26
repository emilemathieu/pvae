import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from pvae.utils import Constants
from pvae.ops.manifold_layers import GeodesicLayer, MobiusLayer, LogZero, ExpZero


def extra_hidden_layer(hidden_dim, non_lin):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), non_lin)


class EncLinear(nn.Module):
    """ Usual encoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncLinear, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)          # flatten data
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class DecLinear(nn.Module):
    """ Usual decoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecLinear, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)


class EncWrapped(nn.Module):
    """ Usual encoder followed by an exponential map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncWrapped, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, manifold.coord_dim)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))
        mu = self.fc21(e)          # flatten data
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class DecWrapped(nn.Module):
    """ Usual encoder preceded by a logarithm map """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecWrapped, self).__init__()
        self.data_size = data_size
        self.manifold = manifold
        modules = []
        modules.append(nn.Sequential(nn.Linear(manifold.coord_dim, hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        z = self.manifold.logmap0(z)
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)


class DecGeo(nn.Module):
    """ First layer is a Hypergyroplane followed by usual decoder """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecGeo, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(GeodesicLayer(manifold.coord_dim, hidden_dim, manifold), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)


class EncMob(nn.Module):
    """ Last layer is a Mobius layers """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso):
        super(EncMob, self).__init__()
        self.manifold = manifold
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(data_size), hidden_dim), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = MobiusLayer(hidden_dim, manifold.coord_dim, manifold)
        self.fc22 = nn.Linear(hidden_dim, manifold.coord_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-len(self.data_size)], -1))            # flatten data
        mu = self.fc21(e)          # flatten data
        mu = self.manifold.expmap0(mu)
        return mu, F.softplus(self.fc22(e)) + Constants.eta,  self.manifold


class DecMob(nn.Module):
    """ First layer is a Mobius Matrix multiplication """
    def __init__(self, manifold, data_size, non_lin, num_hidden_layers, hidden_dim):
        super(DecMob, self).__init__()
        self.data_size = data_size
        modules = []
        modules.append(nn.Sequential(MobiusLayer(manifold.coord_dim, hidden_dim, manifold), LogZero(manifold), non_lin))
        modules.extend([extra_hidden_layer(hidden_dim, non_lin) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, prod(data_size))

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], *self.data_size)  # reshape data
        return mu, torch.ones_like(mu)


class DecBernouilliWrapper(nn.Module):
    """ Wrapper for Bernoulli likelihood """
    def __init__(self, dec):
        super(DecBernouilliWrapper, self).__init__()
        self.dec = dec

    def forward(self, z):
        mu, _ = self.dec.forward(z)
        return torch.tensor(1.0).to(z.device), mu
