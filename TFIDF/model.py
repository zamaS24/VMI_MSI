import torch.nn as nn


import torch
import torch.nn as nn


class MLPNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=2,
        dropout_input=0.35,
        dropout_hidden=0.45,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout_input),

            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_hidden),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_hidden),

            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)
