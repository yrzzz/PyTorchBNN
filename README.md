# Biostat821-Final-Project
## Group member
Ruizhi Yuan, Tianbei (Tiya) Zhang, Jingwen Deng

## Introduction
BayesianNeuralNetwork is a simple PyTorch library for creating Bayesian Neural Network Layers. You can add uncertainty and gather the complexity cost of your model in a simple way that does not affect the interaction between your layers by using BayesianNeuralNetwork layers and utils. You can extend and improve this library by using our core weight sampler classes to add uncertainty to a larger scope of layers in a PyTorch-integrated manner.

## Documentation

Documentation for bayesian layers, losses, weight sampler and utils:
 * [Bayesian Layer](doc/layer.md)
 * [Weight samplers](doc/weight_sampler.md)
 * [Utils (variational inference for bayesian deep learning)](doc/utils.md)
 * [Losses](doc/losses.md)

## A example for classification
This example briefly show how to use this library. The example will train a Bayesian Neural Network using MNIST dataset, and conduct a classification task. 

## Import modules from Pytorch and BayesianNeuralNetwork
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from BayesianNeuralNetwork.modules import BayesianLinear
from BayesianNeuralNetwork.utils import variational_estimator_set

import torchvision.datasets as dsets
import torchvision.transforms as transforms
```

## Load data
```python
train_dataset = dsets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)

test_dataset = dsets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=True
)
```
## Create bayesian variational regressor class
One good thing of this library is that the linear bayesian layer can be conveniently combined with other functions in Pytorch. Linear bayesian layer can be used as easily as torch.nn.linear, and can be add in any layer of a neural network. Actication function can also be applied on bayesian layer. In this example, we creat a simple bayesian multilayer perceptron. Also, one can also create a bayesian deep learning model using ```torch.nn.Sequential()``` such as ```nn.Sequential(BayesianLinear(10, 5), BayesianLinear(5, 1))```

Normally, created neural network class should be decorated by variational_estimator_set so that one can use variational inference methods to facilitate the calculation of bayesian deep learning.
```python
@variational_estimator_set
class BayesianMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.bayesian_linear1 = BayesianLinear(input_dim, 512)
        self.linear = nn.Linear(512, 256)
        self.bayesian_linear2 = BayesianLinear(256, output_dim)

    def forward(self, x):
        x_ = x.view(-1, 28 * 28)
        x_ = self.bayesian_linear1(x_)
        x_ = F.relu(x_)
        x_ = self.linear(x_)
        x_ = F.relu(x_)
        return self.bayesian_linear2(x_)
```
## Create bayesian neural network
Like the normal neural network model, one can use loss function and optimizer given in the Pytorch to help us train the model. And gpu training are supported
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = BayesianMLP(28 * 28, 10).to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
```
## Training and evaluating loop
One thing special in bayesian neural network is that the variational inference methods should be used when calculating the loss. That is because there are kl divergence used during the training, which cannot be directly conduct backward propogation. Variational inference methods can solve this problem.
```python
iteration = 0
for epoch in range(100):
    for i, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        loss = classifier.sample_elbo(
            inputs=features.to(device),
            labels=labels.to(device),
            criterion=criterion,
            sample_nbr=3,
            complexity_cost_weight=1 / 50000,
        )
        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 250 == 0:
            print(loss)
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    outputs = classifier(images.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()
            print(
                "Iteration: {} | Accuracy of the network on the 10000 test images: {} %".format(
                    str(iteration), str(100 * correct / total)
                )
            )
```
