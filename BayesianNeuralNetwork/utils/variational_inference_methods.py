import torch
from BayesianNeuralNetwork.losses import kl_divergence


def variational_estimator_set(model: torch.nn.Module):
    """
    This function will be use as a decorator, which will include several methods that are useful in solving some
    calculation problems in Bayesian Neural Network.

    :param model: a class inherit from torch.nn.Module
    :return: a torch.nn.Module with methods：
                1. Calculate the KL divergence of Bayesian layers in the module
                2. Sample Elbo loss (methond of variational inference)
    """

    def kl_divergence_of_nn(self):
        """
        Calculates the sum of the KL divergence of Bayesian layers in the model, which are a method to estimate the
        similarity of two distribution
        """
        return kl_divergence(self)

    setattr(model, "kl_divergence_of_nn", kl_divergence_of_nn)

    def sample_elbo(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: torch.nn.Module,
        sample_nbr: int,
        complexity_cost_weight: int = 1,
    ):

        """
        Elbo is a method to minimize the kl divergence
        :param inputs: input data of the model
        :param labels: labels of the data
        :param criterion: loss function
        :param sample_nbr: number of times of the weight-sampling and predictions done in Monte-Carlo simulation to
                            gather the loss to be .backwarded in the optimization of the model.
        :return: evidence lower bound
        """

        loss = 0
        for _ in range(sample_nbr):
            outputs = self(inputs)
            loss += criterion(outputs, labels)
            loss += self.kl_divergence_of_nn() * complexity_cost_weight
        return loss / sample_nbr

    setattr(model, "sample_elbo", sample_elbo)

    def mfvi_forward(self, inputs, sample_nbr=10):
        """
        Performs mean-field variational inference, another variational inference method  for the Bayesian Neural Network
        :param inputs: input data
        :param sample_nbr: number of forward passes to be done on the data
        :return: Estimated kl divergence
        """

        result = torch.stack([self(inputs) for _ in range(sample_nbr)])
        return result.mean(dim=0), result.std(dim=0)

    setattr(model, "mfvi_forward", mfvi_forward)

    return model
