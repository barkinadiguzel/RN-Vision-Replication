import torch
import torch.nn as nn
from backbone.encoder import CNNEncoder
from objects.object_encoder import ObjectEncoder
from relation.rn import RelationNetwork

class RNVisionModel(nn.Module):
    """
    Image -> CNN -> Objects -> RN -> Output
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.encoder = CNNEncoder()
        self.object_encoder = ObjectEncoder(add_coords=True)

        object_dim = 24 + 2  
        self.rn = RelationNetwork(object_dim=object_dim,
                                  relation_dim=256,
                                  output_dim=num_classes)

    def forward(self, x):
        features = self.encoder(x)
        objects = self.object_encoder(features)
        out = self.rn(objects)
        return out
