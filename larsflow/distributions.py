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


class FactorizedResampledGaussian(nf.distributions.BaseDistribution):
    """
    Resampled Gaussian factorized over second dimension,
    i.e. first non-batch dimension; can be class-conditional
    """
    def __init__(self, shape, a, T, eps, affine_shape=None, flows=[],
                 group_dim=0, same_dist=True, num_classes=None):
        """
        Constructor
        :param shape: Shape of the variables (after mapped through the flows)
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        :param affine_shape: Shape of the affine layer serving as mean and
        standard deviation; if None, no affine transformation is applied
        :param flows: Flows to be applied after sampling from base distribution
        :param group_dim: Int or list of ints; Dimension(s) to be used for group
        formation; dimension after batch dim is 0
        :param same_dist: Flag; if true, the distribution of each of the groups
        is the same
        :param num_classes: Number of classes in the class-conditional case;
        if None, the distribution is not class conditional
        """
        super().__init__()
        # Write parameters to object
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.a = a
        self.T = T
        self.eps = eps
        self.flows = nn.ModuleList(flows)
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        if isinstance(group_dim, int):
            group_dim = [group_dim]
        self.group_dim = group_dim
        self.group_sum_dim = list(range(1, len(self.group_dim) + 1))
        self.group_shape = []
        self.not_group_shape = []
        for i, s in enumerate(self.shape):
            if i in self.group_dim:
                self.group_shape += [s]
            else:
                self.not_group_shape += [s]
        # Get permutation indizes to form groups
        self.perm = []
        for i in range(self.n_dim):
            if i in self.group_dim:
                self.perm = [i + 1] + self.perm
            else:
                self.perm = self.perm + [i + 1]
        self.perm = [0] + self.perm
        self.perm_inv = [0] * len(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i
        self.same_dist = same_dist
        if same_dist:
            self.num_groups = 1
            self.not_group_prod = np.prod(self.not_group_shape)
        else:
            self.num_groups = np.prod(self.not_group_shape)
        # Normalization constant
        if self.class_cond:
            self.register_buffer("Z", -torch.ones(self.num_classes
                                                  * self.num_groups))
        else:
            self.register_buffer("Z", -torch.ones(self.num_groups))
        # Affine transformation
        self.affine_shape = affine_shape
        if self.affine_shape is None:
            self.affine_transform = None
        elif self.class_cond:
            self.affine_transform = nf.flows.CCAffineConst(self.affine_shape,
                                                           self.num_classes)
        else:
            self.affine_transform = nf.flows.AffineConstFlow(self.affine_shape)

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

    def log_prob(self, z, y=None):
        # Get batch size
        batch_size = z.size(0)
        # Reverse flows
        log_p = 0
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_p = log_p + log_det
        # Reverse affine transform
        if self.affine_transform is not None:
            if self.class_cond:
                z, log_det = self.affine_transform.inverse(z, y)
            else:
                z, log_det = self.affine_transform.inverse(z)
            log_p = log_p + log_det
        # Get Gaussian density
        log_p_gauss = - 0.5 * self.d * np.log(2 * np.pi) \
                      - torch.sum(0.5 * torch.pow(z, 2), dim=self.sum_dim)
        # Update normalization constant
        if self.training or torch.any(self.Z < 0.):
            eps = torch.randn(batch_size, *self.group_shape, dtype=z.dtype,
                              device=z.device)
            acc_ = self.a(eps)
            Z_batch = torch.mean(acc_, dim=0)
            if torch.any(self.Z < 0.):
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        # Get values of a
        z = z.permute(*self.perm).contiguous()
        acc = self.a(z.view(-1, *self.group_shape))
        if self.class_cond:
            acc = acc.view(batch_size, -1, self.num_classes, self.num_groups)
            acc = torch.sum(acc * y[:, None, :, None], dim=2)
        else:
            acc = acc.view(batch_size, -1, self.num_groups)
        if self.same_dist:
            acc = acc.view(batch_size, -1)
        else:
            acc = torch.diagonal(acc, dim1=1, dim2=2)
        acc = acc.view(batch_size, *self.not_group_shape)
        # Get normalization constant
        if self.class_cond:
            Z = y @ Z.view(self.num_classes, self.num_groups)
        if self.same_dist:
            Z = Z.view(-1, *([1] * self.n_dim))
        else:
            Z = Z.view(-1, *self.not_group_shape)
        alpha = (1 - Z) ** (self.T - 1)
        log_p_a = torch.sum(torch.log((1 - alpha) * acc / Z + alpha),
                            dim=self.group_sum_dim)
        if self.same_dist:
            log_p_a = log_p_a * self.not_group_prod
        log_p = log_p + log_p_a + log_p_gauss
        return log_p