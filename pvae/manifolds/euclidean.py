import torch
from geoopt.manifolds import Euclidean as EuclideanParent


class Euclidean(EuclideanParent):

    def __init__(self, dim, c=0.):
        super().__init__(1)
        self.register_buffer("dim", torch.as_tensor(dim, dtype=torch.int))
        self.register_buffer("c", torch.as_tensor(c, dtype=torch.get_default_dtype()))

    @property
    def coord_dim(self):
        return int(self.dim)

    @property
    def device(self):
        return self.c.device

    @property
    def zero(self):
        return torch.zeros(1, self.dim).to(self.device)

    def logdetexp(self, x, y, is_vector=False, keepdim=False):
        result = torch.zeros(x.shape[:-1]).to(x)
        if keepdim: result = result.unsqueeze(-1)
        return result

    def expmap0(self, u):
        return u

    def logmap0(self, u):
        return u

    def proju0(self, u):
        return self.proju(self.zero.expand_as(u), u)

    def transp0(self, x, u):
        return self.transp(self.zero.expand_as(u), x, u)

    def lambda_x(self, x, *, keepdim=False, dim=-1):
        return torch.ones_like(x.sum(dim=dim, keepdim=keepdim))
