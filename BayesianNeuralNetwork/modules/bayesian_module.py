from torch import nn


class BayesianModule(nn.Module):
    """
    Create a base class for Bayesian Neural Network, so that other functions can distinguish the Bayesian layer
    """

    def init(self):
        super().__init__()
