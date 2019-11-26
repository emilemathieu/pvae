import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import minimize_scalar
import torch
from torch.autograd import grad
import unittest
from pvae.utils import rexpand
from pvae.distributions.hyperbolic_radius import HyperbolicRadius, cdf_r

colors = sns.color_palette("hls", 8)


class TestHyperbolicRadius(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        torch.manual_seed(1234)

        self.scale = torch.tensor([.2, .3, .5, .8, 1.]).unsqueeze(-1)
        self.shape = self.scale.shape
        self.scale.requires_grad=True
        self.c = .8 * torch.ones(1)
        self.dim = 10
        self.d = HyperbolicRadius(self.dim, self.c, self.scale)

    def test_sample(self):
        self.scale = torch.tensor([.9, 1.]).unsqueeze(-1)
        self.shape = self.scale.shape
        self.c = torch.ones(1)
        self.dim = 2
        self.d = HyperbolicRadius(self.dim, self.c, self.scale)
        N = 10
        xs = self.d.sample(torch.Size([N]))

    def test_cdf(self):
        cdf = self.d.cdf(torch.zeros(self.shape)).float()
        torch.testing.assert_allclose(cdf, torch.zeros(self.shape))
        cdf = self.d.cdf(50*torch.ones(self.shape)).float()
        torch.testing.assert_allclose(cdf, torch.ones(self.shape))

    def ecdf(self, x, grid):
        N = x.size(0)
        xs, _ = torch.sort(x, dim=0)
        ecdf = torch.zeros(torch.Size([grid.size(-1), *x.shape[1:]]))
        for i, thresh in enumerate(grid):
            ecdf[i, :] = (xs < thresh).sum(0)
        ecdf = ecdf / N
        return ecdf.squeeze(-1).transpose(0, 1)

    def test_sample(self):
        N = 100000
        self.d = HyperbolicRadius(self.dim, self.c, torch.tensor([.5, 1.]).unsqueeze(-1))
        x = self.d.sample(torch.Size([N]))
        logp = self.d.log_prob(x)
        
        # Kolmogorov–Smirnov statistic
        grid = torch.linspace(0, 3, steps=100)
        ecdf = self.ecdf(x, grid)
        cdf = self.d.cdf(rexpand(grid, *self.d.scale.size())).squeeze(-1).t()
        diff = (ecdf - cdf).abs().max()
        assert diff < 5e-3

    def test_sample_plot(self):
        N = 10000
        x = self.d.sample(torch.Size([N]))
        xs, _ = torch.unique(x).sort()
        fig, axes = plt.subplots(self.shape[0], 1, sharex=True, figsize=(12, 20))
        for i in range(self.shape[0]):
            ax = axes[i] if type(axes) == np.ndarray else axes
            n, bins, patches = ax.hist(x[:, i].squeeze(-1), density=True, bins=100, color=colors[i])
            y = torch.Tensor(bins).unsqueeze(-1)
            p = HyperbolicRadius(self.dim, self.c, self.d.scale[[i]]).log_prob(y).exp()
            ax.plot(bins, p.data.numpy(), '--')
        fig.savefig('tests/radius_hist_d{}.pdf'.format(self.dim), bbox_inches='tight', transparent=False)

    def test_impl_rep(self):
        torch.manual_seed(1239)
        self.c = torch.tensor([.5])
        self.dim = 5
        self.scale = torch.tensor([0.8, 1.5]).unsqueeze(-1)
        self.scale.requires_grad=True
        self.d = HyperbolicRadius(self.dim, self.c, self.scale)
        N = 10
        x = self.d.rsample(torch.Size([N])).squeeze(0)
        x.backward(torch.ones_like(x))

        u = self.d.cdf(x).squeeze(-1).detach().numpy()
        delta = 1e-4
        scale2 = self.scale.squeeze(-1).detach().numpy() + delta
        scale1 = self.scale.squeeze(-1).detach().numpy() - delta
        def fun(x, u, scale):
            res = np.abs(cdf_r(torch.tensor(x), torch.tensor(scale), self.c, self.dim).detach().numpy() - u)
            if x < 0: res += np.exp(-10*x)
            return res

        grad_approx = torch.zeros(N, self.scale.size(0))
        for i in range(N):
            for j in range(self.scale.size(0)):
                res2 = minimize_scalar(fun, args=(u[i, [j]], scale2[[j]]), tol=1e-7)
                res1 = minimize_scalar(fun, args=(u[i, [j]], scale1[[j]]), tol=1e-7)
                grad_approx[i, j] = float((res2.x - res1.x) / 2 / delta)
        torch.testing.assert_allclose(self.scale.grad.view(-1), grad_approx.sum(0).view(-1), rtol=0.05, atol=.5)

    def test_logZ(self):
        self.c = torch.tensor([.7])
        self.dim = 5
        self.scale = torch.tensor([0.2, 0.5, 1., 1.5]).unsqueeze(-1)
        self.scale.requires_grad = True
        self.d = HyperbolicRadius(self.dim, self.c, self.scale)
        logZ = self.d.log_normalizer
        true_logZ = torch.tensor([-6.62831268665, -1.51213981062, 4.4095754933, 11.8648601183]).unsqueeze(-1)
        torch.testing.assert_allclose(logZ, true_logZ)

        grad_logZ_scale = grad(logZ, (self.scale), grad_outputs=(torch.ones_like(self.scale)))
        grad_logZ_scale = grad_logZ_scale[0]
        eps = 1e-3
        diff = HyperbolicRadius(self.dim, self.c, self.scale+eps).log_normalizer - HyperbolicRadius(self.dim, self.c, self.scale-eps).log_normalizer
        approx_grad_logZ_scale = diff / 2 / eps
        torch.testing.assert_allclose(grad_logZ_scale, approx_grad_logZ_scale)

        self.c = torch.tensor([.9])
        self.dim = 16
        self.scale = torch.tensor([.2, .5, .8, 1.5]).unsqueeze(-1)
        self.scale.requires_grad = True
        self.d = HyperbolicRadius(self.dim, self.c, self.scale)
        logZ = self.d.log_normalizer
        true_logZ = torch.tensor([-10.829589564, 15.90421163, 55.8887896557, 219.5298998]).unsqueeze(-1)
        torch.testing.assert_allclose(logZ, true_logZ)

        grad_logZ_scale = grad(logZ, (self.scale), grad_outputs=(torch.ones_like(self.scale)))
        grad_logZ_scale = grad_logZ_scale[0]
        diff = HyperbolicRadius(self.dim, self.c, self.scale+eps).log_normalizer - HyperbolicRadius(self.dim, self.c, self.scale-eps).log_normalizer
        approx_grad_logZ_scale = diff / 2 / eps
        torch.testing.assert_allclose(grad_logZ_scale, approx_grad_logZ_scale)

    def test_moments(self):
        self.c = torch.tensor([.7])
        self.dim = 5
        self.scale = torch.tensor([0.2, 0.5, 1., 1.5]).unsqueeze(-1)
        self.d = HyperbolicRadius(self.dim, self.c, self.scale)
        means = self.d.mean
        stds = self.d.stddev
        true_means = torch.tensor([0.433607898056, 1.20011844755, 3.42020368377, 7.53105987861]).unsqueeze(-1)
        true_stds = torch.tensor([0.140120759414, 0.38460648107, 0.949379882501, 1.49865023413]).unsqueeze(-1)
        torch.testing.assert_allclose(means, true_means)
        torch.testing.assert_allclose(stds, true_stds)

        self.c = torch.tensor([.9])
        self.dim = 16
        self.scale = torch.tensor([.2, .5, .8, 1.5]).unsqueeze(-1)
        self.d = HyperbolicRadius(self.dim, self.c, self.scale)
        means = self.d.mean
        stds = self.d.stddev
        true_means = torch.tensor([0.865320299218, 3.57019322017, 9.10736146478, 32.0180613104]).unsqueeze(-1)
        true_stds = torch.tensor([0.153446810207, 0.494167390237, 0.799998631679, 1.49999998722]).unsqueeze(-1)
        torch.testing.assert_allclose(means, true_means)
        torch.testing.assert_allclose(stds, true_stds)


if __name__ == '__main__':
        unittest.main()