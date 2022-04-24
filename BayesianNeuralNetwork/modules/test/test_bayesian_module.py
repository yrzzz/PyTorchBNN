import pytest
from torch import nn
from BayesianNeuralNetwork.modules.bayesian_module import BayesianModule


class TestBayesianModule:
    def test_init(self):
        b_module = BayesianModule()
        assert isinstance(b_module, nn.Module)
