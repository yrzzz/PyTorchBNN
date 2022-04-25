import torch
from torch import nn
from torch.nn import functional as F
from BayesianNeuralNetwork.modules.bayesian_module import BayesianModule
from BayesianNeuralNetwork.modules.weight_sampler import (
    TrainablePosteriorDistribution,
    PriorDistribution,
)


class BayesianLinear(BayesianModule):
    """
    This class is a Bayesian Linear layer, which extends standard networks with posterior inference in order to control over-fitting.
    The objective of Bayesian Linear layer is to create a class that can interact with torch nn.Module API and can be chained in nn.Sequential models with other layers in Pytorch
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma_1: float = 0.1,
        prior_sigma_2: float = 0.1,
        prior_pi: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -5.0,
        prior_dist=None,
    ) -> None:
        """
        :param in_features: input feature numbers for Bayesian Linear layer
        :param out_features: output feature numbers for Bayesian Linear layer
        :param prior_sigma_1: prior sigma on the mixture prior distribution 1
        :param prior_sigma_2: prior sigma on the mixture prior distribution 2
        :param prior_pi: pi in the scaled mixture prior
        :param posterior_mu_init: posterior mean for the weight mu init
        :param posterior_rho_init: posterior mean for the weight rho init
        :param prior_dist: prior distribution
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.prior_dist = prior_dist

        # Variational parameters and sample
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(posterior_mu_init, 0.1)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(posterior_rho_init, 0.1)
        )
        self.weight_sampler = TrainablePosteriorDistribution(
            self.weight_mu, self.weight_rho
        )

        self.bias_mu = nn.Parameter(
            torch.Tensor(out_features).normal_(posterior_mu_init, 0.1)
        )
        self.bias_rho = nn.Parameter(
            torch.Tensor(out_features).normal_(posterior_rho_init, 0.1)
        )
        self.bias_sampler = TrainablePosteriorDistribution(self.bias_mu, self.bias_rho)

        # Prior distribution
        self.weight_prior_dist = PriorDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist
        )
        self.bias_prior_dist = PriorDistribution(
            self.prior_pi, self.prior_sigma_1, self.prior_sigma_2, dist=self.prior_dist
        )
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x: torch.Tensor):
        # Sample the weights and bias, forward them

        w = self.weight_sampler.sample()

        b = self.bias_sampler.sample()
        b_log_posterior = self.bias_sampler.log_likelihood_posterior()
        b_log_prior = self.bias_prior_dist.log_likelihood_prior(b)

        # Complexity cost
        self.log_variational_posterior = (
            self.weight_sampler.log_likelihood_posterior() + b_log_posterior
        )
        self.log_prior = self.weight_prior_dist.log_likelihood_prior(w) + b_log_prior

        return F.linear(x, w, b)
