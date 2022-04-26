# Bayesian Neural Network layer
They all inherit from torch.nn.Module
# Index:
  * [BayesianModule](#class-BayesianModule)
  * [BayesianLinear](#class-BayesianLinear)
## class BayesianModule(torch.nn.Module)
### BayesianNeuralNetwork.modules.bayesian_module.BayesianModule()
Create a base class for Bayesian Neural Network, so that other functions can distinguish the Bayesian layer
Inherits from torch.nn.Module

---

## class BayesianLinear
### BayesianNeuralNetwork.modules.BayesianLinear(in_features, out_features, prior_sigma_1 = 0.1, prior_sigma_2 = 0.1, prior_pi = 0.5, posterior_mu_init: float = 0, posterior_rho_init: float = -5.0, prior_dist=None)

Implements the linear bayesian layer. 

Creates weight samplers of the class TrainablePosteriorDistribution for the weights and biases.

Inherits from BayesianModule

#### Parameters:
  * in_features int -> Number nodes of the information to be feedforwarded
  * out_features int -> Number of out nodes of the layer
  * prior_sigma_1 float -> sigma of one of the prior w distributions to mixture
  * prior_sigma_2 float -> sigma of one of the prior w distributions to mixture
  * prior_pi float -> factor to scale the gaussian mixture of the model prior distribution
  * posterior_mu_init float -> posterior mean for the weight mu init
  * posterior_rho_init float -> posterior mean for the weight rho init
  * prior_dist -> torch.distributions.distribution.Distribution corresponding to a prior distribution different than a normal / scale mixture normal; if you pass that, the prior distribution will be that one and prior_sigma1 and prior_sigma2 and prior_pi can be dismissed. - Note that there is a torch issue that may output you logprob as NaN, so beware of the prior dist you are using.
  
#### Methods:
  * forward():
      
      Performs a forward operation with sampled weights.
      
      Returns torch.tensor
      
      Description
      ##### Parameters
       * x - torch.tensor, the features tensor to be forwarded
