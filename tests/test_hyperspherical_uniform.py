import torch
import math
import unittest

from pvae.distributions import HypersphericalUniform

class TestHypersphericalUniform(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        torch.manual_seed(1234)

        self.dim = 8
        self.d = HypersphericalUniform(self.dim)

    def test_sample(self):
        x = self.d.rsample(torch.Size([5]))
        torch.testing.assert_allclose(x.pow(2).sum(-1), torch.ones(torch.Size([*x.shape[:-1]])))

    def test_log_prob(self):
        d = HypersphericalUniform(2)
        x = d.sample(torch.Size([5]))
        logp = d.log_prob(x)
        torch.testing.assert_allclose(logp, - (math.log(4) + math.log(math.pi)))

    def test_rsample(self):
        x = self.d.rsample(torch.Size([5]))
        y = torch.tensor(2., requires_grad=True)
        loss = (x - y).pow(2).sum()
        loss.backward()

if __name__ == '__main__':
        unittest.main()