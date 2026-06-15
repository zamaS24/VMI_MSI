import torch.nn as nn

class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_layers=(64, 32), output_dim=2, dropout_rates=0.2):
        super(MLPNet, self).__init__()
        if isinstance(dropout_rates, float):
            dropout_rates = [dropout_rates] * len(hidden_layers)

        layers = []
        current_dim = input_dim
        for hidden_dim, dropout_rate in zip(hidden_layers, dropout_rates):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
