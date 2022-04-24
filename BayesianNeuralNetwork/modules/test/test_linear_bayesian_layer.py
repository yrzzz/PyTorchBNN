import pytest
import torch
from torch import nn
from BayesianNeuralNetwork.modules import BayesianLinear
from BayesianNeuralNetwork.modules.bayesian_module import BayesianModule


class TestBayesianLinear:
    def test_init_bayesian_linear(self):
        module = BayesianLinear(5, 1)
        assert isinstance(module, BayesianModule)

    def test_forward(self):
        model = nn.Sequential(BayesianLinear(5, 1), nn.ReLU())
        input = torch.ones(1, 5)
        output = model(input)
        assert output.size() == torch.Size([1, 1])