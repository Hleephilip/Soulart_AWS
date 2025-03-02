import torch
import torch.nn as nn
import numpy as np
import os
import time
import logging
import argparse

from dataset.lidarhuman26M_dataset import lidarhuman26MDataset
from dataset.LIP_dataset import LIPDDataset
from dataset.soulart_dataset import SoulartDataset
from dataset.dataloader import make_dataloader
from model_movin import MOVIN
from utils.smpl_utils import get_smpl_adj_matrix, get_smpl_joint_num, get_smpl_skeleton, rotation_6d_to_matrix
from utils.soulart_utils import get_ours_joint_num, get_ours_adj_matrix, get_ours_skeleton
from utils.seed import seed_fix
from utils.train_utils import load_config, save_model, calc_abs_joint, cal_ang, print_log, load_model
from scipy.spatial.transform import Rotation as R
# from util.loss_utils import kl_div, smooth_l1_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
L1_loss = torch.nn.L1Loss()
use_augment = False

def kl_div(mu, log_std):
    kl = -log_std - 0.5 + (torch.exp(2 * log_std) + mu ** 2) * 0.5
    kl = kl.sum(1).mean()
    return kl

def loss_movin(x_t, g_t, pred_x_t, pred_g_t, mu, log_sigma):
    # l_rec_1 = L1_loss(pred_x_t[:, 1:], x_t[:, 1:])
    l_rec_1 = L1_loss(pred_x_t, x_t)

    # for loss (discarding root rotation)
    _fk_pred_x_t = calc_abs_joint(pred_x_t[:, :, :3], pred_x_t[:, :, 3:], get_ours_skeleton())
    _fk_x_t = calc_abs_joint(x_t[:, :, :3], x_t[:, :, 3:], get_ours_skeleton())
    l_rec_2 = L1_loss(_fk_pred_x_t, _fk_x_t)

    # for metric
    # fk_pred_x_t = calc_abs_joint(pred_x_t[:, :, :3], torch.cat((pred_g_t[:, 3:].unsqueeze(1), pred_x_t[:, 1:, 3:]), axis=1), get_smpl_skeleton())
    # fk_x_t = calc_abs_joint(x_t[:, :, :3], torch.cat((g_t[:, 3:].unsqueeze(1), x_t[:, 1:, 3:]), axis=1), get_smpl_skeleton())
    
    l_rec_3 = L1_loss(pred_g_t, g_t)
    l_kl = kl_div(mu, log_sigma)
    return l_rec_1 + l_rec_2 + l_rec_3 + l_kl, _fk_pred_x_t, _fk_x_t

def train_step(net, dataloader, criterion, epoch, optimizer, report_interval, logging, normalized):

    net.train()

    epoch_log = {'loss': [], 'mpjpe': [], 'abs': [], 'ang': []}
    total_frames = 0
        
    # Training loop
    torch.cuda.synchronize() 

    for i, item in enumerate(dataloader):
        torch.cuda.synchronize() 
        # load variables
        pc = item['pc'].to(torch.float32).to(device) #* 1000 # convert to mm
        x_t = item['x_t'].to(device) # [B, 24, 9]
        x_t_prev = item['x_t_prev'].to(device)
        g_t = item['g_t'].to(device) # [B, 9]
        g_t_prev = item['g_t_prev'].to(device)
        cetroid = item['cetroid'].to(device)
        scale = item['scale'].to(device).unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]

        # Augmentation
        if use_augment:
            # Shuffle pcs
            permutation = torch.randperm(pc.size(1))
            pc = pc[:, permutation, :]

            shift = ((torch.rand((pc.size(0), 3)).to(device) * 40. - 20.) / max_len_3d).view(-1, 1, 3)
            pc[:, :, 0:3] = pc[:, :, 0:3] + shift
            g_t[:, 0:3] = g_t[:, 0:3] + shift
            
            aug_scale = (torch.rand(pc.size(0)).to(device) * 0.4 + 0.8).view(-1, 1, 1)
            pc = pc * aug_scale
            x_t[:, :, 0:3] = x_t[:, :, 0:3] * aug_scale
            g_t[:, 0:3] = g_t[:, 0:3] * aug_scale
            # print(x_t.shape) # [B, 24, 9]

        # Model inference
        optimizer.zero_grad() 
        pred_x_t, pred_g_t, mu, log_sigma = net(pc, x_t, g_t, x_t_prev, g_t_prev)

        # Optimize the model
        loss, abs_pred_x_t, abs_x_t = criterion(x_t, g_t, pred_x_t, pred_g_t, mu, log_sigma)
        loss.backward()
        optimizer.step()
        
        '''
        # compute error in world cs      
        outputs_xyz = pred_pos_t.view(-1, dataset.num_joints * 3)
        gt_xyz = joint_pos_t.view(-1, dataset.num_joints * 3)
        diff = torch.pow(outputs_xyz - gt_xyz, 2).view(-1, dataset.num_joints, 3)
        diff_sum = torch.sum(diff, 2)
        diff_sum_sqrt = torch.sqrt(diff_sum)
        diff_mean = torch.mean(diff_sum_sqrt,1).view(-1,1)
        MPJPE = torch.mul(diff_mean, dataset.curr_max_len_3d).mean() # * 1000 # batch avg 
        '''
        
        B = abs_x_t.shape[0]
        J = x_t.shape[1]
        if normalized:
            # then denormalize
            abs_x_t = abs_x_t.detach().clone() * scale + cetroid.unsqueeze(1)
            abs_pred_x_t = abs_pred_x_t.detach().clone() * scale + cetroid.unsqueeze(1)
        loss_MPJPE = np.linalg.norm(abs_x_t.cpu().detach().numpy() - abs_pred_x_t.cpu().detach().numpy(),axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))
        loss_ABS = np.linalg.norm((abs_x_t + g_t[:, :3].unsqueeze(1) * scale).cpu().detach().numpy() - (abs_pred_x_t + pred_g_t[:, :3].unsqueeze(1) * scale).cpu().detach().numpy(), axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))

        gt_pose = x_t[:, :, 3:] # B, J, 6
        gt_pose[:, 0] = g_t[:, 3:]
        gt_pose = R.from_matrix(rotation_6d_to_matrix(gt_pose.view(B,J,6)).view(-1,3,3).cpu().detach().numpy()).as_rotvec().reshape(-1, J*3)
        pose = pred_x_t[:, :, 3:] # B, J, 6
        pose[:, 0] = pred_g_t[:, 3:]
        pose = R.from_matrix(rotation_6d_to_matrix(pose.view(B,J,6)).view(-1,3,3).cpu().detach().numpy()).as_rotvec().reshape(-1, J*3)
        loss_ang = cal_ang(gt_pose, pose)
        # loss_jitter = jitter(pre_j.reshape(B,T,24,3))

        epoch_log['loss'].append(loss.item() * len(pc))
        epoch_log['mpjpe'].append(loss_MPJPE.item() * len(pc))
        epoch_log['abs'].append(loss_ABS.item() * len(pc))
        epoch_log['ang'].append(loss_ang.item() * len(pc))
        total_frames += len(pc)

        # print loss info
        if i % report_interval == 0:
            log_text = f'Epoch {epoch} Step {i} Train loss {loss.item():.4f}  MPJPE {loss_MPJPE.item()*1000:.2f}mm  ang {loss_ang.item():.2f}deg'
            print_log(log_text, logging)

    torch.cuda.synchronize() 

    train_log = {
        'loss': np.sum(epoch_log['loss']) / total_frames,
        'mpjpe': np.sum(epoch_log['mpjpe']) / total_frames,
        'abs': np.sum(epoch_log['abs']) / total_frames,
        'ang': np.sum(epoch_log['ang']) / total_frames,
        # 'esti_time': epoch_time
    }
    return train_log


def val_step(net, dataloader, criterion, epoch, report_interval, logging, normalized):
    net.eval()

    epoch_log = {'loss': [], 'mpjpe': [], 'abs': [], 'ang': []}
    total_frames = 0

    with torch.no_grad():
        # Validation loop
        torch.cuda.synchronize() 

        for i, item in enumerate(dataloader):
            torch.cuda.synchronize() 

            # load variables
            pc = item['pc'].to(torch.float32).to(device) #* 1000 # convert to mm
            x_t = item['x_t'].to(device) # [B, 24, 9]
            x_t_prev = item['x_t_prev'].to(device)
            g_t = item['g_t'].to(device) # [B, 9]
            g_t_prev = item['g_t_prev'].to(device)
            cetroid = item['cetroid'].to(device)
            scale = item['scale'].to(device).unsqueeze(-1).unsqueeze(-1) # [B, 1, 1]

            # Model inference
            pred_x_t, pred_g_t, mu, log_sigma = net(pc, x_t, g_t, x_t_prev, g_t_prev, validate=True)

            loss, abs_pred_x_t, abs_x_t = criterion(x_t, g_t, pred_x_t, pred_g_t, mu, log_sigma)
            B = abs_x_t.shape[0]
            J = x_t.shape[1]
            if normalized:
                # then denormalize
                abs_x_t = abs_x_t.detach().clone() * scale + cetroid.unsqueeze(1)
                abs_pred_x_t = abs_pred_x_t.detach().clone() * scale + cetroid.unsqueeze(1)
            loss_MPJPE = np.linalg.norm(abs_x_t.cpu().detach().numpy() - abs_pred_x_t.cpu().detach().numpy(),axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))
            loss_ABS = np.linalg.norm((abs_x_t + g_t[:, :3].unsqueeze(1) * scale).cpu().detach().numpy() - (abs_pred_x_t + pred_g_t[:, :3].unsqueeze(1) * scale).cpu().detach().numpy(), axis=2).mean(axis=1).mean()#(loss_fn(out.float(),seq_gt.float()))

            gt_pose = x_t[:, :, 3:] # B, J, 6
            gt_pose[:, 0] = g_t[:, 3:]
            gt_pose = R.from_matrix(rotation_6d_to_matrix(gt_pose.view(B,J,6)).view(-1,3,3).cpu().detach().numpy()).as_rotvec().reshape(-1, J*3)
            pose = pred_x_t[:, :, 3:] # B, J, 6
            pose[:, 0] = pred_g_t[:, 3:]
            pose = R.from_matrix(rotation_6d_to_matrix(pose.view(B,J,6)).view(-1,3,3).cpu().detach().numpy()).as_rotvec().reshape(-1, J*3)
            loss_ang = cal_ang(gt_pose, pose)

            epoch_log['loss'].append(loss.item() * len(pc))
            epoch_log['mpjpe'].append(loss_MPJPE.item() * len(pc))
            epoch_log['abs'].append(loss_ABS.item() * len(pc))
            epoch_log['ang'].append(loss_ang.item() * len(pc))
            total_frames += len(pc)

            # print loss info
            if i % report_interval == 0:
                log_text = f'Epoch {epoch} Step {i} Validation loss {loss.item():.4f}  MPJPE {loss_MPJPE.item()*1000:.2f}mm  ang {loss_ang.item():.2f}deg'
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
    
def train(cfg):
    if cfg.train_dataset.name == "lidarhuman26M":
        train_dataset = lidarhuman26MDataset(cfg.train_dataset)
    elif cfg.train_dataset.name == "LIPD":
        train_dataset = LIPDDataset(cfg.train_dataset)
    elif cfg.train_dataset.name == "Soulart":
        train_dataset = SoulartDataset(cfg.train_dataset)
    else:
        raise NotImplementedError()
    train_dataloader = make_dataloader(train_dataset, batch_size=cfg.exp.batch_size, shuffle=True)

    
    if cfg.val_dataset.name == "lidarhuman26M":
        val_dataset = lidarhuman26MDataset(cfg.val_dataset)
    elif cfg.val_dataset.name == "LIPD":
        val_dataset = LIPDDataset(cfg.val_dataset)
    elif cfg.val_dataset.name == "Soulart":
        val_dataset = SoulartDataset(cfg.val_dataset)
    else:
        raise NotImplementedError()
    val_dataloader = make_dataloader(val_dataset, batch_size=cfg.exp.batch_size, shuffle=False)

    print(len(train_dataloader))
    print(len(val_dataloader))
    
    net = MOVIN(
        in_chans = cfg.model.in_chans,
        out_chans = cfg.model.out_chans,
        num_joints = get_ours_joint_num(),
        adj_matrix = get_ours_adj_matrix().to(device)
    ).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.exp.lr)
    loss_fn = loss_movin
    
    
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename=os.path.join(cfg.exp.path, cfg.exp.title, 'train.log'), level=logging.INFO)
    print_log(f"[Experiment Name] {cfg.exp.title}", logging)

    best_metric = 999.99
    start_epoch = 0

    if cfg.exp.resume_ckpt != "None":
        ckpt = load_model(cfg.exp.resume_ckpt)
        net.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_metric = ckpt['log']['mpjpe']
        print_log("Resuming from the saved checkpoint", logging)

    for epoch in range(start_epoch, cfg.exp.epochs):
        is_best = False
        log = train_step(
            net=net,
            dataloader=train_dataloader, 
            criterion=loss_fn,
            epoch=epoch,
            optimizer=optimizer,
            report_interval=cfg.exp.report_interval,
            logging=logging,
            normalized=True
        )
        log_text = f"[Epoch {epoch}] Train loss {log['loss']:.4f}  MPJPE {log['mpjpe']*1000:.4f}mm  ABS {log['abs']*1000:.4f}mm  ang {log['ang']:.4f}deg"
        print_log(log_text, logging)

        log = val_step(
            net=net, 
            dataloader=val_dataloader, 
            criterion=loss_fn,
            epoch=epoch, 
            report_interval=cfg.exp.report_interval,
            logging=logging,
            normalized=True
        )
        log_text = f"[Epoch {epoch}] Validation loss {log['loss']:.4f}  MPJPE {log['mpjpe']*1000:.4f}mm  ABS {log['abs']*1000:.4f}mm  ang {log['ang']:.4f}deg"
        print_log(log_text, logging)

        save_model(epoch, net, optimizer, os.path.join(cfg.exp.path, cfg.exp.title, "ckpt"), log, "ckpt-last", logging)
        if log['mpjpe'] < best_metric:
            print_log("------------------------- New Record -------------------------", logging)
            best_metric = log['mpjpe']
            save_model(epoch, net, optimizer, os.path.join(cfg.exp.path, cfg.exp.title, "ckpt"), log, "ckpt-best", logging)
        
        # save checkpoints of last 10 epochs
        if epoch > cfg.exp.epochs - 10:
            save_model(epoch, net, optimizer, os.path.join(cfg.exp.path, cfg.exp.title, "ckpt"), log, f"ckpt-epoch-{epoch}", logging)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to the config yaml file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # cfg = load_config("./cfgs/dataset_cfgs/lidarhuman26M.yaml")
    seed_fix(cfg.exp.seed)
    os.makedirs(os.path.join(cfg.exp.path, cfg.exp.title), exist_ok=True)
    os.makedirs(os.path.join(cfg.exp.path, cfg.exp.title, "ckpt"), exist_ok=True)
    train(cfg)
    # test()

if __name__ == '__main__':
    main()
    