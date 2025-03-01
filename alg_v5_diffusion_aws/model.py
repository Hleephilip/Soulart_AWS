import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils.config import *
from utils.smpl_utils import get_smpl_adj_matrix
from models.point_attention import AttnPointEnc
from models.imu_attention import AttnIMUEnc
from models.denoising import Denoiser, Regressor, Xregressor
from models.gcn import x1_mapper
from models.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, mode='cosine'):
        super().__init__()
        assert mode in ('linear', "cosine", "real_linear")
        self.num_steps = num_steps

        self.mode = mode

        t = torch.linspace(0, 1, steps=num_steps+1)
        if mode == 'linear':
            self.log_snr = self.beta_linear_log_snr
        elif mode == "cosine":
            self.log_snr = self.alpha_cosine_log_snr


    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(0, 1), batch_size)
        return ts.tolist()

    def log_snr_to_alpha_sigma(self, log_snr):
        return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

    def beta_linear_log_snr(self, t):
        return -torch.log(torch.special.expm1(1e-4 + 10 * (t ** 2)))

    def alpha_cosine_log_snr(self, t, s: float = 0.008):
        return -torch.log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1)

    def real_linear_beta_schedule(self, timesteps):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = 1 - x / timesteps
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)


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
        self.var_sched = VarianceSchedule(num_steps=self.num_sample_steps, mode='cosine')
    
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


        batch_size = pts.shape[0]
        t = torch.zeros((batch_size,), device=curr_xs.device).float().uniform_(0, 1)
        log_snr = self.var_sched.log_snr(t)
        alpha, sigma = self.var_sched.log_snr_to_alpha_sigma(t)
        c0 = alpha.view(-1, 1, 1)   # (B, 1, 1)
        c1 = sigma.view(-1, 1, 1)   # (B, 1, 1)
        e_rand = torch.randn_like(curr_xs)  # (B, d, J)
        x_t = c0 * curr_xs + c1 * e_rand
        # t, x_t = self.sample_latent(x1 = curr_xs) # B, / B, J, 3
        cond_k_x = torch.cat((f_p, f_imu), dim = -1)
        x1 = self.x_denoiser(x_t, log_snr, cond_k_x) # B, J, 3


        
        t = torch.zeros((batch_size,), device=curr_rs.device).float().uniform_(0, 1)
        log_snr = self.var_sched.log_snr(t)
        alpha, sigma = self.var_sched.log_snr_to_alpha_sigma(t)
        c0 = alpha.view(-1, 1, 1)   # (B, 1, 1)
        c1 = sigma.view(-1, 1, 1)   # (B, 1, 1)
        e_rand = torch.randn_like(curr_rs)  # (B, d, J)
        r_t = c0 * curr_rs + c1 * e_rand
        f_x1 = self.x1_enc(x1) # B, J, 256
        cond_k_r = torch.cat((cond_k_x, f_x1), dim = -1)
        # t, r_t = self.sample_latent(x1 = curr_rs) # B, / B, J, 6
        r1 = self.r_denoiser(r_t, log_snr, cond_k_r)

        return x1, r1

    def sample(self, pts, xs, rs):
        B, _, J, C_x = xs.shape # B, T-1, J, 3
        C_r = rs.shape[-1] # 6
        
        times = torch.linspace(1, 0, steps=self.num_sample_steps + 1, device=xs.device)
        times = times.unsqueeze(0).repeat(B, 1)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        time_pairs = times.unbind(dim=-1)
        
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
        cond_k_x = torch.cat((f_p, f_imu), dim = -1)
        for time, time_next in time_pairs:
            log_snr = self.var_sched.log_snr(time)
            x1_pred = self.x_denoiser(xt, log_snr, cond_k_x)

            log_snr_next = self.var_sched.log_snr(time_next)
            alpha, sigma = self.var_sched.log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = self.var_sched.log_snr_to_alpha_sigma(log_snr_next)

            pred_noise = (xt - alpha.view(-1, 1, 1) * x1_pred) / sigma.view(-1, 1, 1).clamp(min=1e-8)
            xt = x1_pred * alpha_next.view(-1, 1, 1) + pred_noise * sigma_next.view(-1, 1, 1)
            
        # Sample r
        rt = torch.randn(size=(B, J, C_r)).to(device=rs.device)
        f_x1 = self.x1_enc(x1_pred) # B, J, 256
        cond_k_r = torch.cat((cond_k_x, f_x1), dim = -1)
        
        for time, time_next in time_pairs:
            log_snr = self.var_sched.log_snr(time)
            r1_pred = self.r_denoiser(rt, log_snr, cond_k_r)

            log_snr_next = self.var_sched.log_snr(time_next)
            alpha, sigma = self.var_sched.log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = self.var_sched.log_snr_to_alpha_sigma(log_snr_next)

            pred_noise = (rt - alpha.view(-1, 1, 1) * r1_pred) / sigma.view(-1, 1, 1).clamp(min=1e-8)
            rt = r1_pred * alpha_next.view(-1, 1, 1) + pred_noise * sigma_next.view(-1, 1, 1)

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