import torch
import torch.nn as nn

class ObjectEncoder(nn.Module):
    def __init__(self, add_coords=True):
        super().__init__()
        self.add_coords = add_coords

    def forward(self, feature_map):
        B, C, H, W = feature_map.shape

        # flatten spatial grid
        objects = feature_map.view(B, C, H * W)
        objects = objects.permute(0, 2, 1)  # (B, N, C)

        if self.add_coords:
            coords = self._build_coords(B, H, W, feature_map.device)
            objects = torch.cat([objects, coords], dim=-1)

        return objects  # (B, N, C+2)

    def _build_coords(self, B, H, W, device):
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)

        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1)  # (H, W, 2)
        coords = coords.view(1, H * W, 2).repeat(B, 1, 1)

        return coords
