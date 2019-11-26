import math
import torch
from torch.autograd import Function, grad
import torch.distributions as dist
from pvae.utils import Constants, logsinh, log_sum_exp_signs, rexpand
from numbers import Number
from pvae.distributions.ars import ARS


def cdf_r(value, scale, c, dim):
    value = value.double()
    scale = scale.double()
    c = c.double()

    if dim == 2:
        return 1 / torch.erf(c.sqrt() * scale / math.sqrt(2)) * .5 * \
    (2 * torch.erf(c.sqrt() * scale / math.sqrt(2)) + torch.erf((value - c.sqrt() * scale.pow(2)) / math.sqrt(2) / scale) - \
        torch.erf((c.sqrt() * scale.pow(2) + value) / math.sqrt(2) / scale))
    else:
        device = value.device

        k_float = rexpand(torch.arange(dim), *value.size()).double().to(device)
        dim = torch.tensor(dim).to(device).double()

        s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
            + torch.log( \
                torch.erf((value - (dim - 1 - 2 * k_float) * c.sqrt() * scale.pow(2)) / scale / math.sqrt(2)) \
                + torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)) \
                )
        s2 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
            + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)))

        signs = torch.tensor([1., -1.]).double().to(device).repeat(((int(dim)+1) // 2)*2)[:int(dim)]
        signs = rexpand(signs, *value.size())

        S1 = log_sum_exp_signs(s1, signs, dim=0)
        S2 = log_sum_exp_signs(s2, signs, dim=0)

        output = torch.exp(S1 - S2)
        zero_value_idx = value == 0.
        output[zero_value_idx] = 0.
        return output.float()


def grad_cdf_value_scale(value, scale, c, dim):
    device = value.device

    dim = torch.tensor(int(dim)).to(device).double()

    signs = torch.tensor([1., -1.]).double().to(device).repeat(((int(dim)+1) // 2)*2)[:int(dim)]
    signs = rexpand(signs, *value.size())
    k_float = rexpand(torch.arange(dim), *value.size()).double().to(device)

    log_arg1 = (dim - 1 - 2 * k_float).pow(2) * c * scale * \
    (\
        torch.erf((value - (dim - 1 - 2 * k_float) * c.sqrt() * scale.pow(2)) / scale / math.sqrt(2)) \
        + torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)) \
    )
    
    log_arg2 = math.sqrt(2 / math.pi) * ( \
        (dim - 1 - 2 * k_float) * c.sqrt() * torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) \
        - ((value / scale.pow(2) + (dim - 1 - 2 * k_float) * c.sqrt()) * torch.exp(-(value - (dim - 1 - 2 * k_float) * c.sqrt() * scale.pow(2)).pow(2) / (2 * scale.pow(2)))) \
        )

    log_arg = log_arg1 + log_arg2
    sign_log_arg = torch.sign(log_arg)

    s = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
            + torch.log(sign_log_arg * log_arg)

    log_grad_sum_sigma = log_sum_exp_signs(s, signs * sign_log_arg, dim=0)
    grad_sum_sigma = torch.sum(signs * sign_log_arg * torch.exp(s), dim=0)

    s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
        + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
        + torch.log( \
            torch.erf((value - (dim - 1 - 2 * k_float) * c.sqrt() * scale.pow(2)) / scale / math.sqrt(2)) \
            + torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)) \
        )

    S1 = log_sum_exp_signs(s1, signs, dim=0)
    grad_log_cdf_scale = grad_sum_sigma / S1.exp()
    log_unormalised_prob = - value.pow(2) / (2 * scale.pow(2)) + (dim - 1) * logsinh(c.sqrt() * value) - (dim - 1) / 2 * c.log()
    
    with torch.autograd.enable_grad():
        scale = scale.float()
        logZ = _log_normalizer_closed_grad.apply(scale, c, dim)
        grad_logZ_scale = grad(logZ, scale, grad_outputs=torch.ones_like(scale))

    grad_log_cdf_scale = - grad_logZ_scale[0] + 1 / scale + grad_log_cdf_scale.float()
    cdf = cdf_r(value.double(), scale.double(), c.double(), int(dim)).float().squeeze(0)
    grad_scale = cdf * grad_log_cdf_scale

    grad_value = (log_unormalised_prob.float() - logZ).exp()
    return grad_value, grad_scale


class _log_normalizer_closed_grad(Function):
    @staticmethod 
    def forward(ctx, scale, c, dim):
        scale = scale.double()
        c = c.double()
        ctx.scale = scale.clone().detach()
        ctx.c = c.clone().detach()
        ctx.dim = dim

        device = scale.device
        output = .5 * (Constants.logpi - Constants.log2) + scale.log() -(int(dim) - 1) * (c.log() / 2 + Constants.log2)
        dim = torch.tensor(int(dim)).to(device).double()

        k_float = rexpand(torch.arange(int(dim)), *scale.size()).double().to(device)
        s = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
            + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)))
        signs = torch.tensor([1., -1.]).double().to(device).repeat(((int(dim)+1) // 2)*2)[:int(dim)]
        signs = rexpand(signs, *scale.size())
        ctx.log_sum_term = log_sum_exp_signs(s, signs, dim=0)
        output = output + ctx.log_sum_term

        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()

        device = grad_input.device
        scale = ctx.scale
        c = ctx.c
        dim = torch.tensor(int(ctx.dim)).to(device).double()

        k_float = rexpand(torch.arange(int(dim)), *scale.size()).double().to(device)
        signs = torch.tensor([1., -1.]).double().to(device).repeat(((int(dim)+1) // 2)*2)[:int(dim)]
        signs = rexpand(signs, *scale.size())

        log_arg = (dim - 1 - 2 * k_float).pow(2) * c * scale * (1+torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2))) + \
            torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) * 2 / math.sqrt(math.pi) * (dim - 1 - 2 * k_float) * c.sqrt() / math.sqrt(2)
        log_arg_signs = torch.sign(log_arg)
        s = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
            + torch.log(log_arg_signs * log_arg)
        log_grad_sum_sigma = log_sum_exp_signs(s, log_arg_signs * signs, dim=0)

        grad_scale = torch.exp(log_grad_sum_sigma - ctx.log_sum_term)
        grad_scale = 1 / ctx.scale + grad_scale

        grad_scale = (grad_input * grad_scale.float()).view(-1, *grad_input.shape).sum(0)
        return (grad_scale, None, None)


class impl_rsample(Function):
    @staticmethod
    def forward(ctx, value, scale, c, dim):
        ctx.scale = scale.clone().detach().double().requires_grad_(True)
        ctx.value = value.clone().detach().double().requires_grad_(True)
        ctx.c = c.clone().detach().double().requires_grad_(True)
        ctx.dim = dim
        return value

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_cdf_value, grad_cdf_scale = grad_cdf_value_scale(ctx.value, ctx.scale, ctx.c, ctx.dim)
        assert not torch.isnan(grad_cdf_value).any()
        assert not torch.isnan(grad_cdf_scale).any()
        grad_value_scale = -(grad_cdf_value).pow(-1) * grad_cdf_scale.expand(grad_input.shape)
        grad_scale = (grad_input * grad_value_scale).view(-1, *grad_cdf_scale.shape).sum(0)
        # grad_value_c = -(grad_cdf_value).pow(-1) * grad_cdf_c.expand(grad_input.shape)
        # grad_c = (grad_input * grad_value_c).view(-1, *grad_cdf_c.shape).sum(0)
        return (None, grad_scale, None, None)


class HyperbolicRadius(dist.Distribution):
    support = dist.constraints.positive
    has_rsample = True

    def __init__(self, dim, c, scale, ars=True, validate_args=None):
        self.dim = dim
        self.c = c
        self.scale = scale
        self.device = scale.device
        self.ars = ars
        if isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()
        self.log_normalizer = self._log_normalizer()
        if torch.isnan(self.log_normalizer).any() or torch.isinf(self.log_normalizer).any():
            print('nan or inf in log_normalizer', torch.cat((self.log_normalizer, self.scale), dim=1))
            raise
        super(HyperbolicRadius, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        value = self.sample(sample_shape)
        return impl_rsample.apply(value, self.scale, self.c, self.dim)

    def sample(self, sample_shape=torch.Size()):
        if sample_shape == torch.Size(): sample_shape=torch.Size([1])
        with torch.no_grad():
            mean = self.mean
            stddev = self.stddev
            if torch.isnan(stddev).any(): stddev[torch.isnan(stddev)] = self.scale[torch.isnan(stddev)]
            if torch.isnan(mean).any(): mean[torch.isnan(mean)] = ((self.dim - 1) * self.scale.pow(2) * self.c.sqrt())[torch.isnan(mean)]
            steps = torch.linspace(0.1, 3, 10).to(self.device)
            steps = torch.cat((-steps.flip(0), steps))
            xi = [mean + s * torch.min(stddev, .95 * mean / 3) for s in steps]
            xi = torch.cat(xi, dim=1)
            ars = ARS(self.log_prob, self.grad_log_prob, self.device, xi=xi, ns=20, lb=0)
            value = ars.sample(sample_shape)
        return value

    def __while_loop(self, logM, proposal, sample_shape):
        shape = self._extended_shape(sample_shape)
        r, bool_mask = torch.ones(shape).to(self.device), (torch.ones(shape) == 1).to(self.device)
        count = 0
        while bool_mask.sum() != 0:
            count += 1
            r_ = proposal.sample(sample_shape).to(self.device)
            u = torch.rand(shape).to(self.device)
            log_ratio = self.log_prob(r_) - proposal.log_prob(r_) - logM
            accept = log_ratio > torch.log(u)
            reject = 1 - accept
            r[bool_mask * accept] = r_[bool_mask * accept]
            bool_mask[bool_mask * accept] = reject[bool_mask * accept]
        return r

    def log_prob(self, value):
        res = - value.pow(2) / (2 * self.scale.pow(2)) + (self.dim - 1) * logsinh(self.c.sqrt() * value) \
            - (self.dim - 1) / 2 * self.c.log() - self.log_normalizer#.expand(value.shape)
        assert not torch.isnan(res).any()
        return res

    def grad_log_prob(self, value):
        res = - value / self.scale.pow(2) + (self.dim - 1) * self.c.sqrt() * torch.cosh(self.c.sqrt() * value) / torch.sinh(self.c.sqrt() * value) 
        return res

    def cdf(self, value):
        return cdf_r(value, self.scale, self.c, self.dim)

    @property
    def mean(self):
        c = self.c.double()
        scale = self.scale.double()
        dim = torch.tensor(int(self.dim)).double().to(self.device)
        signs = torch.tensor([1., -1.]).double().to(self.device).repeat(((self.dim+1) // 2)*2)[:self.dim].unsqueeze(-1).unsqueeze(-1).expand(self.dim, *self.scale.size())
        
        k_float = rexpand(torch.arange(self.dim), *self.scale.size()).double().to(self.device)
        s2 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
                + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
                + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)))
        S2 = log_sum_exp_signs(s2, signs, dim=0)

        log_arg = (dim - 1 - 2 * k_float) * c.sqrt() * scale.pow(2) * (1 + torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2))) + \
                torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) * scale * math.sqrt(2 / math.pi)
        log_arg_signs = torch.sign(log_arg)
        s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
                + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
                + torch.log(log_arg_signs * log_arg)
        S1 = log_sum_exp_signs(s1, signs * log_arg_signs, dim=0)

        output = torch.exp(S1 - S2)
        return output.float()

    @property
    def variance(self):
        c = self.c.double()
        scale = self.scale.double()
        dim = torch.tensor(int(self.dim)).double().to(self.device)
        signs = torch.tensor([1., -1.]).double().to(self.device).repeat(((int(dim)+1) // 2)*2)[:int(dim)].unsqueeze(-1).unsqueeze(-1).expand(int(dim), *self.scale.size())

        k_float = rexpand(torch.arange(self.dim), *self.scale.size()).double().to(self.device)
        s2 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
                + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
                + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)))
        S2 = log_sum_exp_signs(s2, signs, dim=0)

        log_arg = (1 + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2)) * (1 + torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2))) + \
               (dim - 1 - 2 * k_float) * c.sqrt() * torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) * scale * math.sqrt(2 / math.pi)
        log_arg_signs = torch.sign(log_arg)
        s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
                + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
                + 2 * scale.log() \
                + torch.log(log_arg_signs * log_arg)
        S1 = log_sum_exp_signs(s1, signs * log_arg_signs, dim=0)

        output = torch.exp(S1 - S2)
        output = output.float() - self.mean.pow(2)
        return output

    @property
    def stddev(self): return self.variance.sqrt()

    def _log_normalizer(self): return _log_normalizer_closed_grad.apply(self.scale, self.c, self.dim)
