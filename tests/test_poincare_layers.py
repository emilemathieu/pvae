import torch
import math
import unittest
import numpy as np

from pvae.ops.poincare_layers import *
from pvae.ops.mobius_poincare import exp_map_x

class TestPoincareLayers(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        torch.manual_seed(1234)

    def test_hypergyroplane(self):
        latent_dim = 2
        hidden_dim = 5
        linear_layer = Hypergyroplane(latent_dim, hidden_dim)
        inputs = torch.tensor([[.5, 0.], [0., .1], [-.2, -.2]])
        outputs = linear_layer(inputs)
   
    def test_sign_hypergyroplane(self):
        '''
        Given a Poincaré hyperplane parametrized by a normal a and a bias p, the function test_sign_hypergyroplane maps p
        in the directions a and -a and tests that the outputs of the forward function of the class Hypergyroplane for each 
        point are opposite numbers which absolute values are equal. This function also checks that the absolute value of
        each output of the forward function is equal to :math:`\\text{lambda}_x(p)*\\text{norm}(a)`.
    
        '''
        c = torch.tensor(1.)
        latent_dim = 2
        hidden_dim = 1
        linear_layer = Hypergyroplane(latent_dim, hidden_dim)
        a = linear_layer.weight
        p = linear_layer.get_bias()
        x = exp_map_x(p, a, c)
        y = exp_map_x(p, -a, c)
        inputs = torch.cat((x, y), 0)
        res = linear_layer(inputs)
        self.assertEqual(res[0].item(), -res[1].item())

if __name__ == '__main__':
    unittest.main()