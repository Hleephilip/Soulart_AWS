import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from pointnet2_ops.pointnet2_modules import PointnetSAModule

LEAKY_RATE = 0.1

class PointNet2(nn.Module):
    def __init__(self, out_chans, use_xyz=True):
        super().__init__()

        self.out_chans = out_chans
        self.use_xyz = use_xyz

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[0, 64, 64, 128],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=self.use_xyz
            )
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, self.out_chans),
        )


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))
        # return features

        

class GCN1d(nn.Module):
    def __init__(self, in_channels, out_channels, graph_a, graph_width, use_pool, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=False, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_pool = use_pool

        self.graph_a = graph_a.expand(-1, self.in_channels, -1, -1) # nn.Parameter(torch.randn(1, in_channels, graph_width, graph_width).cuda(), requires_grad=True)
        self.graph_w = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        )
        self.pool = nn.AvgPool1d(graph_width, stride=1)

    def forward(self, x):  # x.shape: B, 3, J
        point1_graph = self.graph_w(torch.matmul(x.unsqueeze(-2), self.graph_a).squeeze(-2)) # [B, 256, J]
        if self.use_pool:
            out = self.pool(point1_graph).squeeze(-1) # [B, 256]
            return out
        return point1_graph.transpose(1, 2).contiguous() # B, J, out_chans


# MLP 
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels)
        )

    def forward(self, x):
        return self.layers(x)



# Transformer Encoder
class GaussianTransformerEnc(nn.Module):
    def __init__(self, in_chans):
        super().__init__()
        self.in_chans = in_chans
        
        # Define learnable tokens as learnable parameters
        self.mu_sigma_tokens = nn.Parameter(torch.randn(2, self.in_chans), requires_grad=True)

        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(d_model=self.in_chans, nhead=4, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=2)
        
    
    def forward(self, feature_vector):
        # Concatenate mu_sigma_tokens and feature_vector
        mu_sigma_tokens_expanded = self.mu_sigma_tokens.expand(feature_vector.size(0), -1, -1)  # Expand to match batch size -> [B, 2, C]
        x = torch.cat((mu_sigma_tokens_expanded, feature_vector), dim=1)
        
        # print(mu_sigma_tokens_expanded.shape, feature_vector.shape)
        # Transformer encoder
        # print(self.mu_sigma_tokens)
        x = self.transformer_encoder(x)
        
        mu = x[:, 0:1] # B, C
        log_sigma = x[:, 1:2] # B, C
        
        return mu, log_sigma

if __name__ == "__main__":
    net = PointNet2(True).to('cuda')
    p = torch.randn(size=(100, 200, 3)).to('cuda')
    print(net(p).shape)