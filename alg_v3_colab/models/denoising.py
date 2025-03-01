import os
import sys
import torch
import torch.nn as nn
import torch.functional as F

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cross_attention import TransformerDecoderLayer

class Xregressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cond_in_chans = config.cond_in_chans # chans of f_p + chans of imu embedding (512 + 256)
        self.use_leaky = config.use_leaky

        self.regress_1 = nn.Linear(self.cond_in_chans, 256)
        self.regress_2 = nn.Linear(256, 64)
        self.regress_3 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(p = 0.2)
        self.relu = nn.ReLU(inplace=True) if not self.use_leaky else nn.LeakyReLU(0.1, inplace=True)


    def forward(self, cond_k):

        '''
        joints_t: joints [B, J, imu_chans = C]
        t: timestep (t \in [0, 1])
        f_p: point cloud embedding [B, C1 = 512]
        f_x: IMU embedding [B, J, C2 = 256]
        '''
        x = self.relu(self.regress_1(cond_k))
        x = self.relu(self.dropout(self.regress_2(x)))
        x = self.regress_3(x) # B, J, self.imu_chans

        return x


class Regressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_joints = config.n_joints
        self.imu_chans = config.imu_chans # 3 (for xyz coord), 6 (for rotation - 6d vec)
        self.mlp = config.mlp
        self.cond_in_chans = config.cond_in_chans # chans of f_p + chans of imu embedding (512 + 256)
        self.n_head = config.n_head
        self.qkv_chans = config.qkv_chans # 256
        self.use_gcn = config.use_gcn
        self.use_leaky = config.use_leaky

        self.mlp_ffns = nn.ModuleList()
        self.graphs_a = nn.ParameterList()
        self.graphs_w = nn.ModuleList()

        _last_chans = self.imu_chans
        for out_channel in self.mlp:
            self.mlp_ffns.append(nn.Linear(_last_chans + 3, out_channel)) # 3 stands for joint position embedding (3)
            if self.use_gcn:
                self.graphs_a.append(nn.Parameter(torch.randn(1, out_channel, self.n_joints, self.n_joints), requires_grad = True))
                self.graphs_w.append(nn.Sequential(
                        nn.Conv1d(out_channel, out_channel, kernel_size = 1, stride = 1, padding = 0, bias = True),
                        nn.ReLU(inplace=True) if not self.use_leaky else nn.LeakyReLU(0.1, inplace = True)
                    )
                )
            _last_chans = out_channel

        self.fc_q = nn.Linear(self.mlp[-1], self.qkv_chans)
        self.fc_k = nn.Linear(self.cond_in_chans, self.qkv_chans)
        self.cross_attn = TransformerDecoderLayer(
            d_model = self.qkv_chans,
            nhead = self.n_head,
            cross_only = True
        )

        self.regress_1 = nn.Linear(self.qkv_chans, 64)
        self.regress_2 = nn.Linear(64, 32)

        self.regress_3 = nn.Linear(32 * self.n_joints, 256)
        self.regress_4 = nn.Linear(256, 128)
        self.regress_5 = nn.Linear(128, self.imu_chans)
        self.dropout = nn.Dropout(p = 0.4)
        self.relu = nn.ReLU(inplace=True) if not self.use_leaky else nn.LeakyReLU(0.1, inplace=True)


    def forward(self, joints_t, cond_k):

        '''
        joints_t: joints [B, J, imu_chans = C]
        t: timestep (t \in [0, 1])
        f_p: point cloud embedding [B, C1 = 512]
        f_x: IMU embedding [B, J, C2 = 256]
        '''
        x = joints_t # B, J, C = self.imu_chans
        B, J, C = x.shape
        assert J == self.n_joints
        assert C == self.imu_chans
        joint_idx = torch.linspace(0, 1, steps=self.n_joints).to(x.device) # (J)

        joint_idx = joint_idx.unsqueeze(0).repeat(B, 1).unsqueeze(-1) # (B, J, 1)
        joint_emb = torch.cat([joint_idx, torch.sin(joint_idx), torch.cos(joint_idx)], dim=-1) # (B, J, 3)
        
        for i, ffn in enumerate(self.mlp_ffns):
            x = torch.cat((x, joint_emb), -1) # B, J, C + 3
            x = self.relu(ffn(x)) # B, J, mlp[i]
            if self.use_gcn:
                x = x.transpose(1, 2).contiguous() # B, mlp[i], J
                x = self.graphs_w[i](torch.matmul(x.unsqueeze(-2), self.graphs_a[i]).squeeze(-2)) # B, mlp[i], J
                x = x.transpose(1, 2).contiguous() # B, J, mlp[i]

        q = self.fc_q(x) # B, J, qkv_chans
        q = q.transpose(1, 2).contiguous() # B, qkv_chans, J
        
        k = self.fc_k(cond_k) # B, J, qkv_chans
        k = k.transpose(1, 2).contiguous() # B, qkv_chans, J
        
        v = self.cross_attn(query = q, key = k, query_pos = None, key_pos = None)
        x = v.transpose(1, 2).contiguous() # B, J, qkv_chans

        x = self.relu(self.regress_1(x))
        x = self.regress_2(x)
        x = x.view(B, -1) # B, J, 32 -> B, J x 32

        x = self.relu(self.regress_3(x))
        x = self.relu(self.dropout(self.regress_4(x)))
        x = self.regress_5(x) # B, J, self.imu_chans

        return x


class Denoiser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_joints = config.n_joints
        self.imu_chans = config.imu_chans # 3 (for xyz coord), 6 (for rotation - 6d vec)
        self.mlp = config.mlp
        self.cond_in_chans = config.cond_in_chans # chans of f_p + chans of imu embedding (512 + 256)
        self.n_head = config.n_head
        self.qkv_chans = config.qkv_chans # 256
        self.use_gcn = config.use_gcn
        self.use_leaky = config.use_leaky
        self.use_residual = config.use_residual

        self.mlp_ffns = nn.ModuleList()
        self.graphs_a = nn.ParameterList()
        self.graphs_w = nn.ModuleList()

        _last_chans = self.imu_chans
        for out_channel in self.mlp:
            self.mlp_ffns.append(nn.Linear(_last_chans + 6, out_channel)) # 6 stands for timestep (3) + joint position embedding (3)
            if self.use_gcn:
                self.graphs_a.append(nn.Parameter(torch.randn(1, out_channel, self.n_joints, self.n_joints), requires_grad = True))
                self.graphs_w.append(nn.Sequential(
                        nn.Conv1d(out_channel, out_channel, kernel_size = 1, stride = 1, padding = 0, bias = True),
                        nn.ReLU(inplace=True) if not self.use_leaky else nn.LeakyReLU(0.1, inplace = True)
                    )
                )
            _last_chans = out_channel

        self.fc_q = nn.Linear(self.mlp[-1], self.qkv_chans)
        self.fc_k = nn.Linear(self.cond_in_chans, self.qkv_chans)
        self.cross_attn = TransformerDecoderLayer(
            d_model = self.qkv_chans,
            nhead = self.n_head,
            cross_only = True
        )

        self.regress_1 = nn.Linear(self.qkv_chans, 256)
        self.regress_2 = nn.Linear(256, 128)
        self.regress_3 = nn.Linear(128, self.imu_chans)
        self.dropout = nn.Dropout(p = 0.4)
        self.relu = nn.ReLU(inplace=True) if not self.use_leaky else nn.LeakyReLU(0.1, inplace=True)


    def forward(self, joints_t, t, cond_k):

        '''
        joints_t: joints [B, J, imu_chans = C]
        t: timestep (t \in [0, 1])
        f_p: point cloud embedding [B, C1 = 512]
        f_x: IMU embedding [B, J, C2 = 256]
        '''
        x = joints_t # B, J, C = self.imu_chans
        B, J, C = x.shape
        assert J == self.n_joints
        assert C == self.imu_chans
        joint_idx = torch.linspace(0, 1, steps=self.n_joints).to(x.device) # (J)

        t = t.view(B, 1) # (B, 1)
        joint_idx = joint_idx.unsqueeze(0).repeat(B, 1).unsqueeze(-1) # (B, J, 1)
        joint_emb = torch.cat([joint_idx, torch.sin(joint_idx), torch.cos(joint_idx)], dim=-1) # (B, J, 3)
        time_emb = torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1)  # (B, 3)
        time_emb = time_emb.unsqueeze(-2).repeat(1, J, 1) # (B, J, 3)
        
        for i, ffn in enumerate(self.mlp_ffns):
            x = torch.cat((time_emb, x, joint_emb), -1) # B, J, C + 6
            x = self.relu(ffn(x)) # B, J, mlp[i]
            if self.use_gcn:
                x = x.transpose(1, 2).contiguous() # B, mlp[i], J
                x = self.graphs_w[i](torch.matmul(x.unsqueeze(-2), self.graphs_a[i]).squeeze(-2)) # B, mlp[i], J
                x = x.transpose(1, 2).contiguous() # B, J, mlp[i]

        q = self.fc_q(x) # B, J, qkv_chans
        q = q.transpose(1, 2).contiguous() # B, qkv_chans, J
        
        k = self.fc_k(cond_k) # B, J, qkv_chans
        k = k.transpose(1, 2).contiguous() # B, qkv_chans, J
        
        v = self.cross_attn(query = q, key = k, query_pos = None, key_pos = None)
        x = v.transpose(1, 2).contiguous() # B, J, qkv_chans

        x = self.relu(self.regress_1(x))
        x = self.relu(self.dropout(self.regress_2(x)))
        x = self.regress_3(x) # B, J, self.imu_chans

        if self.use_residual:
            x = x + joints_t

        return x
