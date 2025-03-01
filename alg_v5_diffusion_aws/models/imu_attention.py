import os
import sys
import torch
import torch.nn as nn

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.gcn import STGCN
from models.cross_attention import TransformerDecoderLayer

class AttnIMUEnc(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.imu_chans = config.imu_chans # 3
        self.temporal_chans = config.temporal_chans # 512
        self.out_chans = config.out_chans # 256
        self.n_head = config.n_head
        self.st_gcn = STGCN(
            in_channels = self.imu_chans,
            use_bn = config.use_bn,
            adjacency_matrix = kwargs['A'],
            edge_importance_weighting = False
        )
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, self.out_chans)
        self.relu = nn.LeakyReLU(0.1, inplace=True)


    def forward(self, f_p_temporal, x):
        # f_p_temporal.shape: B, G, C = 512
        # x.shape: B, T, J, 3
        f_x = self.st_gcn(x) # B, T, J, C = 256
        f_x = torch.mean(f_x, dim=1) # temporal pooling -> B, J, C

        f_x = self.relu(self.fc1(f_x))
        f_x = self.fc2(f_x)

        return f_x  

