import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.config import *
from utils.smpl_utils import get_smpl_adj_matrix
from models.point_attention import AttnPointEnc
from models.imu_attention import AttnIMUEnc
from models.denoising import Denoiser, Regressor, Xregressor
from models.gcn import x1_mapper
from models.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class HPEModel_OT(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.num_sample_steps = config.num_sample_steps

        self.point_enc = AttnPointEnc(self.config.point_encoder)
        self.imu_enc = AttnIMUEnc(self.config.imu_encoder, A = kwargs['A'])
        self.x_denoiser = Denoiser(self.config.x_denoiser)
        self.x1_enc = x1_mapper(self.config.r_denoiser)
        self.r_denoiser = Denoiser(self.config.r_denoiser)
        self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma = 0.0)
    
    def sample_latent(self, x1, x0=None, t=None):
        '''
        x1: Input joint [B, J, 3] 
        x0: Noise [B, J, 3] if None, sampled
        '''

        if x0 is None:
            x0 = torch.randn_like(x1)
        t, xt, _ = self.FM.sample_location_and_conditional_flow(x0, x1, t=t)

        return t, xt  # B / B, J, 3

    def train_iter(self, pts, xs, rs):
        prev_xs = xs[:, :-1, :, :] # B, T-1, J, 3
        curr_xs = xs[:, -1, :, :] # B, J, 3

        prev_rs = rs[:, :-1, :, :] # B, T-1, J, 6
        curr_rs = rs[:, -1, :, :] # B, J, 6

        prev_imus = torch.cat((prev_xs, prev_rs), dim=-1)

        f_p_temporal, f_p = self.point_enc(pts) # B, G, 512 / B, 512
        f_imu = self.imu_enc(f_p_temporal, prev_imus) # B, J, 256
        f_p = f_p.unsqueeze(-2).repeat(1, f_imu.shape[1], 1)

        t, x_t = self.sample_latent(x1 = curr_xs) # B, / B, J, 3
        cond_k_x = torch.cat((f_p, f_imu), dim = -1)
        x1 = self.x_denoiser(x_t, t, cond_k_x) # B, J, 3

        f_x1 = self.x1_enc(x1) # B, J, 256
        cond_k_r = torch.cat((cond_k_x, f_x1), dim = -1)
        t, r_t = self.sample_latent(x1 = curr_rs) # B, / B, J, 6
        r1 = self.r_denoiser(r_t, t, cond_k_r)

        return x1, r1

    def sample(self, pts, xs, rs):
        B, _, J, C_x = xs.shape # B, T-1, J, 3
        C_r = rs.shape[-1] # 6
        timesteps = torch.linspace(0, 1, steps=self.num_sample_steps, device=pts.device)
        timesteps = timesteps.unsqueeze(0).repeat(B, 1)
        timesteps = torch.stack((timesteps[:, :-1], timesteps[:, 1:]), dim=0)
        timesteps = timesteps.unbind(dim=-1)
        
        prev_xs = xs[:, :-1, :, :] # B, T-1, J, 3
        curr_xs = xs[:, -1, :, :] # B, J, 3

        prev_rs = rs[:, :-1, :, :] # B, T-1, J, 6
        curr_rs = rs[:, -1, :, :] # B, J, 6

        prev_imus = torch.cat((prev_xs, prev_rs), dim=-1)

        f_p_temporal, f_p = self.point_enc(pts) # B, G, 512 / B, 512
        f_imu = self.imu_enc(f_p_temporal, prev_imus) # B, J, 256
        f_p = f_p.unsqueeze(-2).repeat(1, f_imu.shape[1], 1)

        # Sample x
        xt = torch.randn(size=(B, J, C_x)).to(device=xs.device)
        x0 = xt.detach().clone()
        cond_k_x = torch.cat((f_p, f_imu), dim = -1)
        for t, tp1 in timesteps:
            x1_pred = self.x_denoiser(xt, t, cond_k_x)
            _, xt = self.sample_latent(x1=x1_pred, x0=x0, t=tp1)
            
        # Sample r
        rt = torch.randn(size=(B, J, C_r)).to(device=rs.device)
        r0 = rt.detach().clone()
        f_x1 = self.x1_enc(x1_pred) # B, J, 256
        cond_k_r = torch.cat((cond_k_x, f_x1), dim = -1)
        for t, tp1 in timesteps:
            r1_pred = self.r_denoiser(rt, t, cond_k_r)
            _, rt = self.sample_latent(x1=r1_pred, x0=r0, t=tp1)

        return x1_pred, r1_pred



def check_implement():
    cfg = load_config("./cfgs/dataset_cfgs/lidarhuman26M.yaml")
    print_config(cfg)
    # print(type(cfg.dataset))
    model = HPEModel_OT(config = cfg.model, A = get_smpl_adj_matrix()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    sample_pts = torch.randn(size=(2, 2, 256, 3), device=device)
    sample_xs = torch.randn(size=(2, 2, 24, 3), device=device)
    sample_rs = torch.randn(size=(2, 2, 24, 6), device=device)
    out_x, out_r, _ = model.train_iter(sample_pts, sample_xs, sample_rs)
    out_sample_x, out_sample_r, _ = model.sample(sample_pts, sample_xs[:, :-1, :, :], sample_rs[:, :-1, :, :])
    print(out_x.shape, out_r.shape) # [128, 24, 3]
    print(out_sample_x.shape, out_sample_r.shape)

    sample_gt = torch.randn(size=(2, 24, 3), device=device)
    loss = torch.mean((sample_gt - out_x) ** 2)
    print(loss)
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), './cfgs/model.pth')

if __name__ == "__main__":
    check_implement()