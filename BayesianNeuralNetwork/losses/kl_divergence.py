from torch import nn
from BayesianNeuralNetwork.modules.bayesian_module import BayesianModule


def kl_divergence(model: nn.Module):

    """
    Gathers the KL Divergence of Bayesian layers in a nn.Module object
    """
    kl_divergence_init = 0
    for module in model.modules():
        if isinstance(module, (BayesianModule)):
            kl_divergence_init += module.log_variational_posterior - module.log_prior
    return kl_divergence_init
