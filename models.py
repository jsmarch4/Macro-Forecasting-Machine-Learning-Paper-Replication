import torch
from torch import nn


class QuantileNetwork(nn.Module):
    def __init__(self, n_features, hidden_layers=0, hidden_dim=8, alpha=1.0):
        super().__init__()

        layers = []
        input_dim = n_features

        for _ in range(hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=alpha))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

    def l2_penalty(self):
        penalty = 0.0

        for name, param in self.named_parameters():
            if "weight" in name:
                penalty = penalty + torch.sum(param ** 2)

        return penalty