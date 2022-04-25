import pytest
import torch
from torch import nn

from BayesianNeuralNetwork.modules import BayesianLinear
from BayesianNeuralNetwork.losses import kl_divergence
from BayesianNeuralNetwork.utils import variational_estimator_set


class TestVariationalInferenceMethods:
    def test_kl_divergence(self):
        input = torch.ones((1, 5))

        @variational_estimator_set
        class nn_model(nn.Module):
            def __init__(self):
                super().__init__()
                self.bayesian_linear = BayesianLinear(5, 5)

            def forward(self, x):
                return self.bayesian_linear(x)

        model = nn_model()
        predicted = model(input)

        complexity_cost = model.kl_divergence_of_nn()
        kl_complexity_cost = kl_divergence(model)

        assert (complexity_cost == kl_complexity_cost).all() == torch.tensor(True)

    def test_sample_elbo(self):
        dat = torch.randn(2, 5)

        @variational_estimator_set
        class nn_model(nn.Module):
            def __init__(self):
                super().__init__()
                self.nn = nn.Sequential(BayesianLinear(5, 5), BayesianLinear(5, 5))

            def forward(self, x):
                return self.nn(x)

        model = nn_model()
        elbo = model.sample_elbo(
            inputs=dat[0],
            labels=dat[1],
            criterion=torch.nn.MSELoss(),
            sample_nbr=5,
            complexity_cost_weight=1,
        )

        elbo = model.sample_elbo(
            inputs=dat[0],
            labels=dat[1],
            criterion=torch.nn.MSELoss(),
            sample_nbr=5,
            complexity_cost_weight=0,
        )

        assert (elbo == elbo).all() == torch.tensor(True)

    def test_mfvi(self):
        @variational_estimator_set
        class nn_model(nn.Module):
            def __init__(self):
                super().__init__()
                self.nn = nn.Sequential(BayesianLinear(10, 5), BayesianLinear(5, 1))

            def forward(self, x):
                return self.nn(x)

        model = nn_model()
        input = torch.ones(2, 10)
        out = model(input)

        mean, std = model.mfvi_forward(input, sample_nbr=3)
        assert out.shape == mean.shape
        assert out.shape == std.shape

