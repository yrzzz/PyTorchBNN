import pytest
import torch
from BayesianNeuralNetwork.modules.weight_sampler import (
    TrainablePosteriorDistribution,
    PriorDistribution,
)


class TestTrainablePosteriorDistribution:
    def test_sample(self):
        mu = torch.Tensor(5, 5).uniform_(-1, 1)
        rho = torch.Tensor(5, 5).uniform_(-1, 1)
        dist = TrainablePosteriorDistribution(mu, rho)
        s1 = dist.sample()
        s2 = dist.sample()
        assert (s1 != s2).any() == torch.tensor(True)
        assert mu.shape == s1.shape
        assert rho.shape == s1.shape

    def test_log_likelihood_posterior(self):
        mu = torch.Tensor(5, 5).uniform_(-1, 1)
        rho = torch.Tensor(5, 5).uniform_(-1, 1)
        dist = TrainablePosteriorDistribution(mu, rho)
        s1 = dist.sample()
        log_posterior = dist.log_likelihood_posterior()
        check = log_posterior == log_posterior
        assert check == torch.tensor(True)


class TestPriorDistribution:
    def test_prior_without_dist(self):
        mu = torch.Tensor(5, 5).uniform_(-1, 1)
        rho = torch.Tensor(5, 5).uniform_(-1, 1)
        dist = TrainablePosteriorDistribution(mu, rho)
        s1 = dist.sample()
        log_posterior = dist.log_likelihood_posterior()
        prior_dist = PriorDistribution(0.5, 1, 0.01)
        log_prior = prior_dist.log_likelihood_prior(s1)
        check1 = log_prior == log_prior
        check2 = log_posterior <= log_posterior - log_prior
        assert check1 == torch.tensor(True)
        assert check2 == torch.tensor(True)

    def test_prior_with_dist(self):
        mu = torch.Tensor(10, 10).uniform_(-1, 1)
        rho = torch.Tensor(10, 10).uniform_(-1, 1)
        dist = TrainablePosteriorDistribution(mu, rho)
        s1 = dist.sample()
        log_posterior = dist.log_likelihood_posterior()
        prior_dist = PriorDistribution(dist=torch.distributions.normal.Normal(0, 1))
        log_prior = prior_dist.log_likelihood_prior(s1)
        check1 = log_prior == log_prior
        check2 = log_posterior <= log_posterior - log_prior
        assert check1 == torch.tensor(True)
        assert check2 == torch.tensor(True)
