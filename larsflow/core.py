import torch
import normflow as nf

class NormalizingFlow(nf.NormalizingFlow):
    """
    Normalizing Flow model to approximate target distribution
    """
    def __init__(self, q0, flows, p=None):
        """
        Constructor
        :param q0: Base distribution
        :param flows: List of flows
        :param p: Target distribution
        """
        super().__init__(q0, flows, p)

    def reverse_kld_cov(self, num_samples=1, beta=1.):
        """
        Estimates reverse KL divergence, gradients through covariance
        :param num_samples: Number of samples to draw from base distribution
        :return: Estimate of the reverse KL divergence averaged over latent samples
        """
        z0, log_q = self.q0(num_samples)
        log_det = torch.tensor(0.)
        z = z0.detach()
        for flow in self.flows:
            z, log_det_part = flow(z)
            log_det = log_det + log_det_part
        log_p = self.p.log_prob(z)
        log_p_ = log_p.detach()
        log_det_ = log_det.detach()
        log_q_ = log_q.detach()
        A = log_q_ - log_det_ - log_p_
        A_ = A - torch.mean(A)
        grad_a = torch.sum(A_ * log_q) / (num_samples - 1)
        grad_f = -(torch.mean(log_p) + torch.mean(log_det))
        rkld = torch.mean(log_q_) - torch.mean(log_p_) - torch.mean(log_det_)
        return rkld + grad_a - grad_a.detach() + beta * (grad_f - grad_f.detach())


class Glow(nf.MultiscaleFlow):
    """
    Glow model with multiscale architecture, see arXiv:1807.03039
    """
    def __init__(self, config):
        """
        Constructor
        :param config: Config dictionary, to be created from yaml file
        """
        # Get parameters
        L = config['levels']
        K = config['blocks']

        input_shape = config['input_shape']
        channels = input_shape[0]
        hidden_channels = config['hidden_channels']
        split_mode = config['split_mode']
        scale = config['scale']
        use_lu = False if not 'use_lu' in config else config['use_lu']
        class_cond = config['class_cond']
        if class_cond:
            num_classes = config['num_classes']
        else:
            num_classes = None

        # Get transform
        if 'transform' in config:
            if config['transform']['type'] == 'logit':
                transform = nf.transforms.Logit(alpha=config['transform']['param'])
            else:
                raise NotImplementedError('The transform ' + config['transform']['type']
                                          + ' is not yet implemented')
        else:
            transform = None

        # Set up flows, distributions and merge operations
        q0 = []
        merges = []
        flows = []
        for i in range(L):
            flows_ = []
            for j in range(K):
                flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                              split_mode=split_mode, scale=scale,
                                              use_lu=use_lu)]
            flows_ += [nf.flows.Squeeze()]
            flows += [flows_]
            if i > 0:
                merges += [nf.flows.Merge()]
                latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i),
                                input_shape[2] // 2 ** (L - i))
            else:
                latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L,
                                input_shape[2] // 2 ** L)
            if config['base']['type'] == 'gauss_channel':
                affine_shape = latent_shape[:1] + ((1,) * (len(latent_shape) - 1))
                q0 += [nf.distributions.AffineGaussian(latent_shape, affine_shape,
                                                       num_classes)]
                #q0 += [nf.distributions.GlowBase(latent_shape, num_classes)]
            else:
                raise NotImplementedError('The base distribution ' + config['base']['type']
                                          + ' is not implemented')

        # Construct flow model
        super().__init__(q0, flows, merges, transform, class_cond)

    def forward(self, x, y=None, autocast=False):
        """
        Forward pass for data parallel computation
        :param x: Input batch
        :param y: Labels of input batch
        :param autocast: Flag whether to do autocast inside forward pass
        (necessary to use mixed precision with DataParallel model)
        :return: Negative log-likelihood of batch
        """
        if autocast:
            with torch.cuda.amp.autocast():
                out = -self.log_prob(x, y)
                return out
        else:
            return -self.log_prob(x, y)