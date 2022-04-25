import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from BayesianNeuralNetwork.losses import kl_divergence
from BayesianNeuralNetwork.modules import BayesianLinear


class TestKLDivergence:

    def test_kl_divergence(self):
        bayesian_linear = BayesianLinear(5, 5)
        input = torch.ones((1, 5))
        predicted = bayesian_linear(input)

        complexity_cost = bayesian_linear.log_variational_posterior - bayesian_linear.log_prior
        kl_complexity_cost = kl_divergence(bayesian_linear)
        check = (complexity_cost == kl_complexity_cost). all()
        assert check == torch.tensor(True)
        pass
