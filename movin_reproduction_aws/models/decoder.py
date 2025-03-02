import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_chans, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_chans)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class GatingNetwork(nn.Module):
    def __init__(self, in_chans, num_experts):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(in_chans, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.gating(x)


class MoEDecoder(nn.Module):
    def __init__(self, in_chans, out_chans, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([Expert(in_chans, out_chans) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(in_chans, num_experts)

    def forward(self, x):
        gating_weights = self.gating_network(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(expert_outputs * gating_weights.unsqueeze(-1), dim=1)
        return output