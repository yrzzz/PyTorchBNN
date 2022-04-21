import torch
import numpy as np
import torch.nn as nn


class RandomDistribution(nn.Module):
    """
        Samples weights for variational inference
        Calculates the variational posterior part for the loss
    """

    def __init__(self, mu: float, rho: float):
        """
        :param mu: the mean for the samples linear transformation parameters
        :param rho: the standard deviation for the samples linear transformation parameters
        """
        super().__init__()

        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        self.pi = np.pi
        self.w = None
        self.sigma = None
        self.register_buffer('w_eps', torch.Tensor(self.mu.shape))

    def sample(self):
        """
        Samples weights form a Normal distribution, multiplies sigma (a function from a trainable parameter), and adds a mean
        Sets sampled weights as the current weights
        :return: torch.tensor with same shape as self.mu and self.rho
        """

        self.w_eps.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.sigma * self.w_eps + self.mu
        return self.w

    def log_likelihood_posterior(self, w=None):
        """
        Calculates the log_likelihood for sampled weights as a part of the loss
        :param w: sampled weights
        :return: log_likelihood_posteriors
        """

        assert (self.w is not None), "If W has already been sampled, you can only have a log posterior for it."
        if w is None:
            w = self.w

        log_likelihood_posteriors = -np.log(np.sqrt(2 * self.pi)) - torch.log(self.sigma) - (((w - self.mu) ** 2) / (2 * self.sigma ** 2)) - 0.5
        return log_likelihood_posteriors.sum()
