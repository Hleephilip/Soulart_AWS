import yaml
import torch
import os
import numpy as np
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation as R
from utils.smpl_utils import get_smpl_skeleton, get_smpl_joint_num, rotation_6d_to_matrix, matrix_to_rotation_6d, matrix_to_quaternion, quaternion_to_axis_angle


def load_config(path):
    with open(path) as fin:
        config = edict(yaml.safe_load(fin))

    return config

def calc_abs_joint(rel_joints, rotations, joint_tree):
    B, J, _ = rel_joints.shape
    # rotations: B, J, 6
    abs_joints = torch.zeros_like(rel_joints) # B, J, 3
    global_rotation_mat = compute_global_rotation_matrix_from_6d(rotations, joint_tree) # B, J, 3, 3

    for j in range(1, J):
        parent = joint_tree[j - 1][0]
        # Compute absolute position: transform relative joint using parent's global rotation
        abs_joints[:, j, :] = torch.einsum("bij, bj -> bi", global_rotation_mat[:, parent], rel_joints[:, j, :]) + abs_joints[:, parent, :]

    return abs_joints


def compute_global_rotation_matrix_from_6d(rel_rotations, joint_tree):
    """
    Computes absolute rotation matrices from parent-relative axis-angle rotations.

    Args:
        rel_rotations: (B, J, 6) tensor of parent-relative rotations in axis-angle.

    Returns:
        global_rotations: (J, 3, 3) tensor of absolute rotation matrices.
    """
    B = rel_rotations.shape[0]
    J = rel_rotations.shape[1]
    global_rotations = torch.zeros((B, J, 3, 3), device=rel_rotations.device)
    global_rotations[:, 0] = rotation_6d_to_matrix(rel_rotations[:, 0])
    
    for j in range(1, J):
        # Convert axis-angle to rotation matrix
        R_rel = rotation_6d_to_matrix(rel_rotations[:, j])
        
        parent = joint_tree[j - 1][0]
        # Compute global rotation: multiply parent's global rotation by relative rotation
        global_rotations[:, j] = global_rotations[:, parent].clone() @ R_rel

    return global_rotations


def save_model(epoch, net, optimizer, path, log, prefix, logging):
    torch.save({
        'log': log,
        'epoch': epoch,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }, f = os.path.join(path, f"{prefix}.pt"))
    print_log(f"Saved checkpoint at {os.path.join(path, f'{prefix}.pt')}", logging)
    return

def print_log(log_txt, logging):
    print(log_txt)
    logging.info(log_txt)
    return

def load_model(path):
    return torch.load(path, map_location='cpu')

# Copied from https://github.com/4DVLab/LiveHPS/blob/main/test.py 
def local2global(pose):
    kin_chains = [
        [20, 18, 16, 13, 9, 6, 3, 0],   # left arm
        [21, 19, 17, 14, 9, 6, 3, 0],   # right arm
        [7, 4, 1, 0],                   # left leg
        [8, 5, 2, 0],                   # right leg
        [12, 9, 6, 3, 0],               # head
        [0],                            # root, hip
    ]
    T = pose.shape[0]
    Rb2l = []
    cache = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    for chain in kin_chains:
        leaf_rotmat = torch.eye(3).unsqueeze(0).repeat(T,1,1)
        for joint in chain:
            joint_rotvec = pose[:, joint*3:joint*3+3]
            joint_rotmat = torch.from_numpy(R.from_rotvec(joint_rotvec.cpu()).as_matrix().astype(np.float32)).to("cpu")
            leaf_rotmat = torch.einsum("bmn,bnl->bml", joint_rotmat, leaf_rotmat)
            cache[joint] = leaf_rotmat
        Rb2l.append(leaf_rotmat)
    return cache

def local2global_soulart(pose):
    kin_chains = [
        [8, 7, 6, 5, 2, 1, 0],     # left arm
        [12, 11, 10, 9, 2, 1, 0],  # right arm
        [16, 15, 14, 13, 0],    # left leg
        [20, 19, 18, 17, 0],    # right leg
        [4, 3, 2, 1, 0],        # head
        [0],                    # root, hip
    ]
    T = pose.shape[0]
    Rb2l = []
    cache = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    for chain in kin_chains:
        leaf_rotmat = torch.eye(3).unsqueeze(0).repeat(T,1,1)
        for joint in chain:
            joint_rotvec = pose[:, joint*3:joint*3+3]
            joint_rotmat = torch.from_numpy(R.from_rotvec(joint_rotvec.cpu()).as_matrix().astype(np.float32)).to("cpu")
            leaf_rotmat = torch.einsum("bmn,bnl->bml", joint_rotmat, leaf_rotmat)
            cache[joint] = leaf_rotmat
        Rb2l.append(leaf_rotmat)
    return cache

# Modified from https://github.com/4DVLab/LiveHPS/blob/main/test.py 
def cal_ang(gt_pose, pose):
    _dim = gt_pose.shape[-1]
    globalR = torch.from_numpy(pose[:, :3]).float()
    gt_matrix = local2global_soulart(torch.from_numpy(gt_pose).reshape(-1,_dim))
    pose_matrix = local2global_soulart(torch.from_numpy(pose).reshape(-1,_dim))
    #print(gt_matrix)
    gt_matrix = torch.from_numpy(np.array([item.cpu().detach().numpy() for item in gt_matrix if item!=None])).reshape(-1,3,3)
    pose_matrix = torch.from_numpy(np.array([item.cpu().detach().numpy() for item in pose_matrix if item!=None])).reshape(-1,3,3)
    gt_axis = quaternion_to_axis_angle(matrix_to_quaternion(gt_matrix))
    pose_axis = quaternion_to_axis_angle(matrix_to_quaternion(pose_matrix))
    #print(gt_axis.shape)
    gt_norm = np.rad2deg(np.linalg.norm(gt_axis.numpy(),axis=1)).reshape((globalR.shape[0],-1,1))
    pose_norm = np.rad2deg(np.linalg.norm(pose_axis.numpy(),axis=1)).reshape((globalR.shape[0],-1,1))
    anger = np.abs((gt_norm-pose_norm)).mean(axis=1).mean()

    return anger