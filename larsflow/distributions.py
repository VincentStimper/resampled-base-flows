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
        self.Z = None
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, self.d))
            self.log_scale = nn.Parameter(torch.zeros(1, self.d))
        else:
            self.register_buffer("loc", torch.zeros(1, self.d))
            self.register_buffer("log_scale", torch.zeros(1, self.d))

    def forward(self, num_samples=1):
        t = 0
        z = torch.zeros(num_samples, self.d, dtype=self.loc.dtype, device=self.loc.device)
        s = 0
        n = 0
        Z_sum = 0
        for i in range(self.T):
            eps = torch.randn((num_samples, self.d), dtype=self.loc.dtype, device=self.loc.device)
            z_ = self.loc + torch.exp(self.log_scale) * eps
            acc = self.a(z_)
            if self.training or self.Z == None:
                Z_sum = Z_sum + torch.sum(acc).detach()
                n = n + num_samples
            dec = torch.rand_like(acc) < acc
            for j, dec_ in enumerate(dec[:, 0]):
                if dec_ or t == self.T - 1:
                    z[s, :] = z_[j, :]
                    s = s + 1
                    t = 0
                else:
                    t = t + 1
                if s == num_samples:
                    break
            if s == num_samples:
                break
        log_p_gauss = - 0.5 * self.d * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(self.log_scale), 2), 1)
        acc = self.a(z)
        if self.training or self.Z == None:
            eps = torch.randn((num_samples, self.d), dtype=self.loc.dtype, device=self.loc.device)
            z_ = self.loc + torch.exp(self.log_scale) * eps
            Z_batch = torch.mean(self.a(z_))
            Z_ = (Z_sum + Z_batch.detach() * num_samples) / (n + num_samples)
            if self.Z == None:
                self.Z = Z_
            else:
                self.Z = ((1 - self.eps) * self.Z + self.eps * Z_).detach()
            self.Z = Z_batch - Z_batch.detach() + self.Z
        alpha = (1 - self.Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / self.Z + alpha) + log_p_gauss
        return z, log_p

    def log_prob(self, z):
        log_p_gauss = - 0.5 * self.d * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(self.log_scale), 2), 1)
        acc = self.a(z)
        if self.training or self.Z == None:
            eps = torch.randn_like(z)
            z_ = self.loc + torch.exp(self.log_scale) * eps
            Z_batch = torch.mean(self.a(z_))
            if self.Z == None:
                self.Z = Z_batch
            else:
                self.Z = ((1 - self.eps) * self.Z + self.eps * Z_batch).detach()
                self.Z = Z_batch - Z_batch.detach() + self.Z
        alpha = (1 - self.Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / self.Z + alpha) + log_p_gauss
        return log_p