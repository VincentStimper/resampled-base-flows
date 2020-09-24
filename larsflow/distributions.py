import torch
import torch.nn as nn
import numpy as np

import normflow as nf

class ResampledGaussian(nf.distributions.BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix,
    resampled according to a acceptance probability determined by a neural network,
    see arXiv 1810.11428
    """
    def __init__(self, d, a, T, eps, trainable=True):
        """
        Constructor
        :param d: Dimension of Gaussian distribution
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        """
        super().__init__()
        self.d = d
        self.a = a
        self.T = T
        self.eps = eps
        self.register_buffer("Z", torch.tensor(-1.))
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, self.d))
            self.log_scale = nn.Parameter(torch.zeros(1, self.d))
        else:
            self.register_buffer("loc", torch.zeros(1, self.d))
            self.register_buffer("log_scale", torch.zeros(1, self.d))

    def forward(self, num_samples=1):
        t = 0
        eps = torch.zeros(num_samples, self.d, dtype=self.loc.dtype, device=self.loc.device)
        s = 0
        n = 0
        Z_sum = 0
        for i in range(self.T):
            eps_ = torch.randn((num_samples, self.d), dtype=self.loc.dtype, device=self.loc.device)
            acc = self.a(eps_)
            if self.training or self.Z < 0.:
                Z_sum = Z_sum + torch.sum(acc).detach()
                n = n + num_samples
            dec = torch.rand_like(acc) < acc
            for j, dec_ in enumerate(dec[:, 0]):
                if dec_ or t == self.T - 1:
                    eps[s, :] = eps_[j, :]
                    s = s + 1
                    t = 0
                else:
                    t = t + 1
                if s == num_samples:
                    break
            if s == num_samples:
                break
        z = self.loc + torch.exp(self.log_scale) * eps
        log_p_gauss = - 0.5 * self.d * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale, 1)\
                      - torch.sum(0.5 * torch.pow(eps, 2), 1)
        acc = self.a(eps)
        if self.training or self.Z < 0.:
            eps_ = torch.randn((num_samples, self.d), dtype=self.loc.dtype, device=self.loc.device)
            Z_batch = torch.mean(self.a(eps_))
            Z_ = (Z_sum + Z_batch.detach() * num_samples) / (n + num_samples)
            if self.Z < 0.:
                self.Z = Z_
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_gauss
        return z, log_p

    def log_prob(self, z):
        eps = (z - self.loc) / torch.exp(self.log_scale)
        log_p_gauss = - 0.5 * self.d * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale, 1) \
                      - torch.sum(0.5 * torch.pow(eps, 2), 1)
        acc = self.a(eps)
        if self.training or self.Z < 0.:
            eps_ = torch.randn_like(z)
            Z_batch = torch.mean(self.a(eps_))
            if self.Z < 0.:
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_gauss
        return log_p