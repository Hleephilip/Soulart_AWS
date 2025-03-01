import torch
import torch.nn as nn
import numpy as np
import os
import time
import logging
import argparse
import open3d as o3d

from dataset.lidarhuman26M_dataset import lidarhuman26MDataset
# from dataset.LIP_dataset import LIPDDataset
from dataset.dataloader import make_dataloader
from model import HPEModel_OT
from utils.smpl_utils import get_smpl_adj_matrix, get_smpl_joint_num, get_smpl_skeleton, rotation_6d_to_matrix
from utils.seed import seed_fix
from utils.train_utils import load_config, save_model, calc_abs_joint, cal_ang, print_log, load_model
from scipy.spatial.transform import Rotation as R
# from util.loss_utils import kl_div, smooth_l1_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
use_augment = False

def smooth_l1_loss(input, target, sigma=10., reduce=True, normalizer=1.0):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduce:
        return torch.sum(loss) / normalizer
    return torch.sum(loss, dim=1) / normalizer


def loss_alg(x, r, abs_x, pred_x, pred_r, pred_abs_x):
    # l_rec_1 = L1_loss(pred_x_t[:, 1:], x_t[:, 1:])
    B = x.shape[0]
    J = x.shape[1]

    l_rec_1 = smooth_l1_loss(pred_x, x) / (B * J)

    l_rec_2 = smooth_l1_loss(pred_r, r) / (B * J)

    l_rec_3 = smooth_l1_loss(pred_abs_x, abs_x) / B

    return l_rec_1 + l_rec_2 + l_rec_3


def test_step(net, dataloader, criterion, report_interval, logging, normalized):
    net.eval()

    epoch_log = {'loss': [], 'mpjpe': [], 'abs': [], 'ang': []}
    total_frames = 0

    with torch.no_grad():
        # Validation loop
        torch.cuda.synchronize() 

        for i, item in enumerate(dataloader):
            torch.cuda.synchronize() 

            # load variables
            pts = item['pts'].to(torch.float32).to(device) # [B, T, N, 3] #* 1000 # convert to mm
            xs = item['xs'].to(device) # [B, T-1, J, 3]
            rs = item['rs'].to(device) # [B, T-1, J, 6]
            x = item['x'].to(device) # B, J, 3
            r = item['r'].to(device) # B, J, 6
            abs_x = item['abs_x'].to(device) # B, 3
            cetroid = item['cetroid'].to(device)
            scale = item['scale'].to(device).unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]

            # Model inference
            pred_x, pred_r, pred_abs_x = net.sample(pts, xs, rs)

            loss = criterion(x, r, abs_x, pred_x, pred_r, pred_abs_x)
            B = x.shape[0]
            if normalized:
                # then denormalize
                x = x.detach().clone() * scale + cetroid.unsqueeze(1)
                pred_x = pred_x.detach().clone() * scale + cetroid.unsqueeze(1)
            loss_MPJPE = np.linalg.norm(x.cpu().detach().numpy() - pred_x.cpu().detach().numpy(),axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))
            loss_ABS = np.linalg.norm((x + abs_x.unsqueeze(1) * scale).cpu().detach().numpy() - (pred_x + pred_abs_x.unsqueeze(1) * scale).cpu().detach().numpy(), axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))

            gt_pose = r # B, J, 6
            gt_pose = R.from_matrix(rotation_6d_to_matrix(gt_pose.view(B,24,6)).view(-1,3,3).cpu().detach().numpy()).as_rotvec().reshape(-1, 72)
            pose = pred_r # B, J, 6
            pose = R.from_matrix(rotation_6d_to_matrix(pose.view(B,24,6)).view(-1,3,3).cpu().detach().numpy()).as_rotvec().reshape(-1, 72)
            loss_ang = cal_ang(gt_pose, pose)
        
            '''
            point_cloud = o3d.geometry.PointCloud()
            # abs_vis_gt = (abs_x_t + g_t[:, :3].unsqueeze(1) * scale).cpu().detach().numpy() 
            # abs_vis_pred = (abs_pred_x_t + pred_g_t[:, :3].unsqueeze(1) * scale).cpu().detach().numpy()
            # point_cloud.points = o3d.utility.Vector3dVector(np.array(pc[0][-1].detach().clone().cpu())) 
            point_cloud.points = o3d.utility.Vector3dVector(x.cpu().detach().numpy()[0]) 
            o3d.io.write_point_cloud('test_ours_gt.ply', point_cloud)

            print(x[0, :, :3])
            print(pred_x[0, :, :3])
            exit(0)
            '''

            # print(abs_x_t, abs_pred_x_t)

            epoch_log['loss'].append(loss.item() * len(pts))
            epoch_log['mpjpe'].append(loss_MPJPE.item() * len(pts))
            epoch_log['abs'].append(loss_ABS.item() * len(pts))
            epoch_log['ang'].append(loss_ang.item() * len(pts))
            total_frames += len(pts)

            # print loss info
            if i % report_interval == 0:
                log_text = f'Step {i} Validation loss {loss.item():.4f}  MPJPE {loss_MPJPE.item()*1000:.2f}mm  ang {loss_ang.item():.2f}deg'
                print_log(log_text, logging)

    torch.cuda.synchronize() 

    val_log = {
        'loss': np.sum(epoch_log['loss']) / total_frames,
        'mpjpe': np.sum(epoch_log['mpjpe']) / total_frames,
        'abs': np.sum(epoch_log['abs']) / total_frames,
        'ang': np.sum(epoch_log['ang']) / total_frames,
        # 'esti_time': epoch_time
    }
    return val_log
    
def test(cfg):
    if cfg.val_dataset.name == "lidarhuman26M":
        val_dataset = lidarhuman26MDataset(cfg.val_dataset)
    elif cfg.val_dataset.name == "LIPD":
        val_dataset = LIPDDataset(cfg.val_dataset)
    else:
        raise NotImplementedError()
    val_dataloader = make_dataloader(val_dataset, batch_size=cfg.exp.batch_size, shuffle=False)

    print(len(val_dataloader))
    
    net = HPEModel_OT(config = cfg.model, A = get_smpl_adj_matrix()).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = loss_alg
    
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=os.path.join(cfg.exp.path, cfg.exp.title, 'train.log'), level=logging.INFO)
    print_log(f"[Experiment Name] {cfg.exp.title}", logging)

    best_metric = 999.99
    start_epoch = 0

    ckpt = load_model(cfg.exp.resume_ckpt)
    net.load_state_dict(ckpt['model'])
    start_epoch = ckpt['epoch'] + 1
    best_metric = ckpt['log']['mpjpe']
    print_log("Loaded the saved checkpoint", logging)

    log = test_step(
        net=net, 
        dataloader=val_dataloader, 
        criterion=loss_fn,
        report_interval=cfg.exp.report_interval,
        logging=logging,
        normalized=True
    )
    log_text = f"Test loss {log['loss']:.4f}  MPJPE {log['mpjpe']*1000:.4f}mm  \
        ABS {log['abs']*1000:.4f}mm  ang {log['ang']:.4f}deg"
    print_log(log_text, logging)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to the config yaml file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # cfg = load_config("./cfgs/dataset_cfgs/lidarhuman26M.yaml")
    seed_fix(cfg.exp.seed)
    test(cfg)
    # test()

if __name__ == '__main__':
    main()
    