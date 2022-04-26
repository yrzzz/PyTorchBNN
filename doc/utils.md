# Variational estimator for bayesian deep learning

# Index:
  * [Variatioanl_inference_methods](#Variatioanl_inference_methods)
---
## Variatioanl_inference_methods

Some util methods to object that inherits from torch.nn.Module in order to conduct variational inference.

### @variational_estimator_set(model)
  #### Parameters:
  * model: -> torch.nn.Module to have introduced the Bayesian DL methods
    
### Methods:
  * #### kl_divergence_of_nn()
    
    Returns torch.tensor corresponding to the summed KL divergence (relative to the curren weight sampling) of all of its Bayesian layers.
    
  * #### sample_elbo(inputs, labels, criterion, sample_nbr)
    
    Samples the ELBO loss of the model sample_nbr times by doing forward operations and summing its model kl divergence with the loss the criterion outputs.
    
    ##### Parameters:
      * inputs: torch.tensor -> the input data to the model
      * labels: torch.tensor -> label data for the performance-part of the loss calculation
      
        The shape of the labels must match the label-parameter shape of the criterion (one hot encoded or as index, if needed)
               
      * criterion: torch.nn.Module, custom criterion (loss) function, torch.nn.functional function -> criterion to gather the performance cost for the model
      * sample_nbr: int -> The number of times of the weight-sampling and predictions done in our Monte-Carlo approach to gather the loss to be .backwarded in the optimization of the model.

    #### Returns:
      * loss: torch.tensor -> elbo loss for the data given

  * #### mfvi_forward(inputs, sample_nbr)

    Performs mean-field variational inference for the variational estimator model on the inputs

    ##### Parameters:
      * inputs: torch.tensor -> the input data to the model
      * sample_nbr: int -> number of forward passes to be done on the data
    ##### Returns:
      * mean_: torch.tensor -> mean of the perdictions along each of the features of each datapoint
      * std_: torch.tensor -> std of the predictions along each of the features of each datapoint



  
