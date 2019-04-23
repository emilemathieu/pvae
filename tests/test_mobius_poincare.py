import torch
import math
import unittest
import numpy as np

from pvae.ops.mobius_poincare import *

class TestMobiusPoincare(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        torch.manual_seed(1234)

    def test_projection(self):
        c = torch.tensor([1.])
        x = torch.randn(5, 3)
        norm_x = norm(x)
        x = x / (1.1*norm_x)
        mask = torch.distributions.Bernoulli(.5).sample(torch.Size([5,1]))
        x = (1 + 1.5 * mask.expand(5, 3)) * x
        y = project_hyp_vecs(x, c)
        self.assertTrue((y.pow(2).sum(-1) <= 1 / c.sqrt()).all())
        
    def test_parallel_transport(self):
        c = torch.tensor([1.])

        v = torch.randn(3)
        x = exp_map_zero(torch.randn(3), c)

        v2 = parallel_transport_x(x, v, c)
        v3 = parallel_transport_0(x, v2, c)
        [self.assertAlmostEqual(v[i], v3[i]) for i in range(len(v))]

        y = exp_map_zero(torch.randn(3), c)
        v2 = parallel_transport_y_x(y, x, v, c)
        v3 = parallel_transport_y_x(x, y, v2, c)
        [self.assertAlmostEqual(v[i], v3[i]) for i in range(len(v))]

if __name__ == '__main__':
        unittest.main()