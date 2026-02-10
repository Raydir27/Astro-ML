import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

def xmm_encoder(encoder_path):
    model = MLPEncoder(in_dim=6, hidden_dim=128, out_dim=64)
    model.load_state_dict(torch.load(encoder_path))
    return model

def gaia_encoder(encoder_path):
    model = MLPEncoder(in_dim=8, hidden_dim=128, out_dim=64)
    model.load_state_dict(torch.load(encoder_path))
    return model