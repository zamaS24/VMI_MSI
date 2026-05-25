import torch.nn as nn


class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_layers=(64, 32), output_dim=2):
        super(MLPNet, self).__init__()
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
