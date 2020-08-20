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
            log_det += log_det_part
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