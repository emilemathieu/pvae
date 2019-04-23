import torch
import math
import unittest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pvae.distributions.riemannian_normal import RiemannianNormal
from pvae.distributions.wrapped_normal import WrappedNormal
from pvae.distributions.hyperbolic_radius import HyperbolicRadius, cdf_r
from pvae.ops.mobius_poincare import exp_map_x_polar, poinc_dist_sq

colors = sns.color_palette("hls", 8)

class TestRiemannianNormal(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        torch.manual_seed(1234)

        self.dim = 3
        self.c = torch.ones(1)
        self.shape = torch.Size([2, self.dim])
        self.loc = torch.tensor([[.5, 0.], [.5, 0.], [0.5, 0.5]])
        self.scale = torch.tensor([[.1], [.3], [.8]])
        self.scale.requires_grad=True
        self.d = RiemannianNormal(self.c, self.loc, self.scale, self.dim)

    def test_sample(self):
        x = self.d.sample()
        logp = self.d.log_prob(x)
        x = self.d.sample(torch.Size([5]))
        logp = self.d.log_prob(x)

    def test_prob(self):
        N = 1000
        x = self.d.sample(torch.Size([N]))
        logp = self.d.log_prob(x)

if __name__ == '__main__':
        unittest.main()