import torch
import numpy as np
import torch.nn as nn
import torch.functional as F


class TrainablePosteriorDistribution(nn.Module):
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
        self.register_buffer("w_eps", torch.Tensor(self.mu.shape))

    def sample(self):
        """
        Samples weights form a Normal distribution, multiplies sigma (a function from a trainable parameter), and adds a mean
        Sets sampled weights as the current weights
        :return: sampled weight
        """

        self.w_eps.data.normal_()
        self.sigma = torch.log1p(torch.exp(self.rho))
        self.w = self.sigma * self.w_eps + self.mu
        return self.w

    def log_likelihood_posterior(self, w=None) -> torch.Tensor:
        """
        Calculates the log_likelihood for sampled weights as a part of the loss
        :param w: sampled weights
        :return: log_likelihood_posteriors
        """
        assert (
            self.w is not None
        ), "If W has already been sampled, you can only have a log posterior for it."
        if w is None:
            w = self.w

        log_posteriors = (
            -np.log(np.sqrt(2 * self.pi))
            - torch.log(self.sigma)
            - (((w - self.mu) ** 2) / (2 * self.sigma ** 2))
            - 0.5
        )
        return log_posteriors.sum()


class PriorDistribution(nn.Module):
    """
    Calculate the scale mixture prior
    """

    def __init__(
        self, pi: float = 1, sigma1: float = 0.1, sigma2: float = 0.001, dist=None
    ):
        super().__init__()

        if dist is None:
            self.pi = pi
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.dist1 = torch.distributions.Normal(0, sigma1)
            self.dist2 = torch.distributions.Normal(0, sigma2)

        else:
            self.pi = 1
            self.dist1 = dist
            self.dist2 = None

    def log_likelihood_prior(self, w: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log likelihood for prior distribution of weights
        :param w: weights
        :return: log likelihood pdf of prior distribution
        """
        prob_n1 = torch.exp(self.dist1.log_prob(w))

        if self.dist2 is not None:
            prob_n2 = torch.exp(self.dist2.log_prob(w))
        else:
            prob_n2 = 0

        prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2) + 1e-6

        return (torch.log(prior_pdf) - 0.5).sum()
