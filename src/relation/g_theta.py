import torch
import torch.nn as nn
import torch.nn.functional as F

class GTheta(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)
