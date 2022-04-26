import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from BayesianNeuralNetwork.modules import BayesianLinear
from BayesianNeuralNetwork.utils import variational_estimator_set

import torchvision.datasets as dsets
import torchvision.transforms as transforms


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = BayesianMLP(28 * 28, 10).to(device)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

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
