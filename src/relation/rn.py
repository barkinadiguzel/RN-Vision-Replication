import torch
import torch.nn as nn
from .g_theta import GTheta
from .f_phi import FPhi

class RelationNetwork(nn.Module):
    def __init__(self, object_dim, relation_dim=256, output_dim=10):
        super().__init__()

        self.g_theta = GTheta(input_dim=object_dim * 2,
                              hidden_dim=relation_dim,
                              output_dim=relation_dim)

        self.f_phi = FPhi(input_dim=relation_dim,
                          hidden_dim=relation_dim,
                          output_dim=output_dim)

    def forward(self, objects):
        B, N, D = objects.shape

        oi = objects.unsqueeze(2).repeat(1, 1, N, 1)
        oj = objects.unsqueeze(1).repeat(1, N, 1, 1)

        pairs = torch.cat([oi, oj], dim=-1)  
        pairs = pairs.view(B, N * N, 2 * D)

        relations = self.g_theta(pairs)     
        relations = relations.sum(dim=1)     

        out = self.f_phi(relations)
        return out
