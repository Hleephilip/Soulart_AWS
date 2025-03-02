import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.encoder import *
from models.decoder import *
from utils.smpl_utils import get_smpl_adj_matrix, get_smpl_joint_num

# reparametrization trick
def reparametrization_trick(mu, log_std):
    z = torch.randn_like(mu)*log_std.exp() + mu
    return z


# Define MOVIN Model
class MOVIN(nn.Module):
    def __init__(self, in_chans, out_chans, num_joints, adj_matrix):
        super().__init__()
        self.in_chans = in_chans # 9
        self.out_chans = out_chans # 1024
        self.num_joints = num_joints # 24
        self.adj_matrix = adj_matrix.unsqueeze(1)

        self.pointnet2 = PointNet2(out_chans=self.out_chans, use_xyz=True)
        # self.pointnet2 = PointNet2_v2()
        self.gcn = GCN1d(in_channels=self.in_chans, out_channels=self.out_chans, graph_a=self.adj_matrix, graph_width=self.num_joints, use_pool=True)
        self.mlp = MLP(in_channels=self.in_chans, out_channels=self.out_chans)  # MLP for global feature
        self.transformer = GaussianTransformerEnc(in_chans=self.out_chans)
        
        self.moe_decoder = MoEDecoder(in_chans=self.out_chans*8, out_chans=self.out_chans*2, num_experts=8)
        self.de_gcn = GCN1d(in_channels=self.out_chans, out_channels=self.in_chans, graph_a=self.adj_matrix, graph_width=self.num_joints, use_pool=False)
        self.de_mlp = MLP(in_channels=self.out_chans, out_channels=self.in_chans)

    def forward(self, pt, xt, gt, xt_prev, gt_prev, validate=False):
        # Encode features
        '''
        # Input features
        - pt: B, T = 5, N, 3
        - xt, xt_prev: B, J, in_chans = 9
        - gt, gt_prev: B, in_chans = 9

        # Outputs
        - x_hat: B, J, in_chans
        - g_hat: B, in_chans
        - mu, log_sigma: B, out_chans 
        '''

        B = pt.shape[0]; T = pt.shape[1]
        _f_p_lst = []
        for t in range(T):
            _f_p = self.pointnet2(pt[:, t, :, :]).unsqueeze(1)  # B, 1, 1024
            _f_p_lst.append(_f_p)
            del _f_p
        f_p = torch.cat(_f_p_lst, dim=1) # B, T, 1024
        f_x = self.gcn(xt.transpose(1,2).contiguous()).unsqueeze(1) # B, 1, 1024
        f_g = self.mlp(gt).unsqueeze(1) # B, 1, 1024
        f_x_prev = self.gcn(xt_prev.transpose(1,2).contiguous()).unsqueeze(1)
        f_g_prev = self.mlp(gt_prev).unsqueeze(1)

        # print(f_p.shape, f_x.shape, f_g.shape)
        # Concatenate and encode through transformer
        embeddings = torch.cat([f_p, f_x_prev, f_g_prev, f_x, f_g], dim=1) # B, 9, 1024
        # print(embeddings.shape)
        mu, log_sigma = self.transformer(embeddings)
        # print(mu.shape, log_sigma.shape)
        if not validate:
            z = reparametrization_trick(mu, log_sigma) # B, 1, 1024
        else:
            z = torch.randn_like(mu)
        # Decode features
        decode_emb = torch.cat([z, f_p, f_x_prev, f_g_prev], dim=1).view(B, -1) # B, 8, 1024 -> B, 8*1024
        f_d = self.moe_decoder(decode_emb) # B, 2 * 1024
        x_hat = self.de_gcn(f_d[:, :self.out_chans].unsqueeze(-1).expand(-1, -1, self.num_joints)) # B, J, 9
        g_hat = self.de_mlp(f_d[:, self.out_chans:]) # B, 9
        return x_hat, g_hat, mu, log_sigma


if __name__ == "__main__":
    # Initialize and Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MOVIN(
        in_chans = 9,
        out_chans = 1024,
        num_joints = get_smpl_joint_num(),
        adj_matrix = get_smpl_adj_matrix().to(device)
    )
    model.to(device)

    B = 32
    pt = torch.randn(size=(B, 5, 250, 3)).to(device)
    xt = torch.randn(size=(B, 24, 9)).to(device) 
    xt_prev = torch.randn(size=(B, 24, 9)).to(device) 
    gt = torch.randn(size=(B, 9)).to(device) 
    gt_prev = torch.randn(size=(B, 9)).to(device) 

    x_hat, g_hat, mu, log_sigma = model(pt, xt, gt, xt_prev, gt_prev)    
    print(x_hat.shape, g_hat.shape) # torch.Size([32, 24, 9]) torch.Size([32, 9])



    # optimizer = AdamW(model.parameters(), lr=1e-4)
    # dataloader = DataLoader(...)  # Placeholder for actual dataset
    # val_dataloader = DataLoader(...)  # Placeholder for validation dataset
    # w_kl = 1.0  # Set hyperparameter for KL loss weight

    # train_model(model, dataloader, optimizer, w_kl, device)
    # validation_loss, validation_mpjpe = validate_model(model, val_dataloader, w_kl, device)
    # print(f"Validation Loss: {validation_loss}, MPJPE: {validation_mpjpe}")
