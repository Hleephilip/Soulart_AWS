import torch
import os
import numpy as np
import pandas as pd
import sys
from plyfile import PlyData
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.soulart_utils import get_ours_skeleton


'''
active_joints_idx = [0, 1, 2, 3, 4, 
                     5, 6, 7, 8, 
                     9, 10, 11, 12, 
                     13, 14, 15, 
                     17, 18, 19]
'''

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def euler_to_matrix(euler_angles):
    """
    Convert batch of Euler angles to rotation matrices.
    euler_angles: array of shape [J, 3], where columns represent rx, ry, rz in radians.
    Output: array of shape [J, 3, 3] (batch of rotation matrices)
    """
    rx, ry, rz = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    J = rx.shape[0]
    
    Rz = np.zeros((J, 3, 3))
    Rz[:, 0, 0] = cz
    Rz[:, 0, 1] = -sz
    Rz[:, 1, 0] = sz
    Rz[:, 1, 1] = cz
    Rz[:, 2, 2] = 1
    
    Ry = np.zeros((J, 3, 3))
    Ry[:, 0, 0] = cy
    Ry[:, 0, 2] = sy
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sy
    Ry[:, 2, 2] = cy
    
    Rx = np.zeros((J, 3, 3))
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cx
    Rx[:, 1, 2] = -sx
    Rx[:, 2, 1] = sx
    Rx[:, 2, 2] = cx
    
    R = np.einsum('ijk,ikl->ijl', Rz, Ry)  # Compute Rz @ Ry for each J
    R = np.einsum('ijk,ikl->ijl', R, Rx)   # Compute (Rz @ Ry) @ Rx for each J
    
    return torch.from_numpy(R)  # Shape: [J, 3, 3]

def matrix_to_axis_angle(matrix: torch.Tensor, fast: bool = False) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    """
    if not fast:
        return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    omegas = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    norms = torch.norm(omegas, p=2, dim=-1, keepdim=True)
    traces = torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1).unsqueeze(-1)
    angles = torch.atan2(norms, traces - 1)

    zeros = torch.zeros(3, dtype=matrix.dtype, device=matrix.device)
    omegas = torch.where(torch.isclose(angles, torch.zeros_like(angles)), zeros, omegas)

    near_pi = torch.isclose(((traces - 1) / 2).abs(), torch.ones_like(traces)).squeeze(
        -1
    )

    axis_angles = torch.empty_like(omegas)
    axis_angles[~near_pi] = (
        0.5 * omegas[~near_pi] / torch.sinc(angles[~near_pi] / torch.pi)
    )

    # this derives from: nnT = (R + 1) / 2
    n = 0.5 * (
        matrix[near_pi][..., 0, :]
        + torch.eye(1, 3, dtype=matrix.dtype, device=matrix.device)
    )
    axis_angles[near_pi] = angles[near_pi] * n / torch.norm(n)

    return axis_angles



def process_csv(csv_path):
    """
    Extracts the specified joints' positions and rotations from the CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        tuple: (positions, rotations), where each is a NumPy array of shape (J, 3).
    """
    df = pd.read_csv(csv_path)
    
    # Indices of joints to extract
    joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Extract positions and rotations for the selected joints
    joints = df.iloc[joint_indices][["PosX", "PosY", "PosZ"]].to_numpy() # J, 3
    rotations = df.iloc[joint_indices][["RotX", "RotY", "RotZ"]].to_numpy() # J, 3
    
    # joints = torch.from_numpy(joints)
    rotations_axis_angle = matrix_to_axis_angle(euler_to_matrix(rotations), fast=True)
    rotations_6d = matrix_to_rotation_6d(euler_to_matrix(rotations))

    return joints, rotations_axis_angle, rotations_6d


def process_csv_old(file):
    f = pd.read_csv(file)
    # print(f)
    # print(len(f))
    # print(f.iloc[1])
    # print(f.iloc[1, 1:4])
    # print(list(f.iloc[1, 1:4]))
    joints = []
    rotations = []

    for i in active_joints_idx:
        j, r = list(f.iloc[i, 1:4]), list(f.iloc[i, 4:7])
        joints.append(j); rotations.append(r)
    
    joints = torch.tensor(joints).to(torch.float32); rotations = torch.tensor(rotations).to(torch.float32)

    # print(joints.shape, rotations.shape)
    # print(joints)
    # print(rotations)
    return joints, rotations


def random_point_sample(pc, n_points = 256):
    N = pc.shape[0]
    if N < n_points:
        dummy_pc = np.zeros(shape=(n_points-N, 3))
        output_pc = np.concatenate((pc, dummy_pc), axis=0)
    elif N > n_points: # two cases: 1) repeat and random sample 2) random sample from first
        indices = np.random.permutation(N)[:n_points]
        output_pc = pc[indices]
    else: # if size is exactly match
        output_pc = pc
    
    return output_pc

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    # pc = pc - centroid
    # _pc = pc - centroid
    # m = np.max(np.sqrt(np.sum(_pc**2, axis=1)))
    # pc = pc / m
    return pc, centroid

def read_point_cloud_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    ply_data = PlyData.read(filename)['vertex'].data
    points = np.array([[x, y, z] for x, y, z in ply_data])
    return points


def compute_global_rotation_matrix_from_axis_angle(rel_rotations, joint_tree):
    """
    Computes absolute rotation matrices from parent-relative axis-angle rotations.

    Args:
        rel_rotations: (J*3) tensor of parent-relative rotations in axis-angle.

    Returns:
        global_rotations: (J, 3, 3) tensor of absolute rotation matrices.
    """
    J = rel_rotations.shape[0] // 3
    global_rotations = torch.zeros((J, 3, 3), device=rel_rotations.device)
    global_rotations[0] = torch.from_numpy(R.from_rotvec(rel_rotations[0:3].cpu()).as_matrix().astype(np.float32)).to("cpu")

    for j in range(1, J):
        # Convert axis-angle to rotation matrix
        R_rel = torch.from_numpy(R.from_rotvec(rel_rotations[j*3:j*3+3].cpu()).as_matrix().astype(np.float32)).to("cpu")
        
        parent = joint_tree[j - 1][0]
        # Compute global rotation: multiply parent's global rotation by relative rotation
        global_rotations[j] = torch.matmul(global_rotations[parent], R_rel)

    return global_rotations

def calc_rel_joint(joint, rotations, joint_tree):
    joint_processed = np.zeros(joint.shape)
    # joint_processed[0] = joint[0]
    for i in range(1, 21):
        parent = joint_tree[i-1][0]
        diff = torch.from_numpy(joint[i] - joint[parent])
        joint_processed[i, :] = np.array(torch.einsum("ij, j -> i", rotations[parent].transpose(-1, -2), diff))

    # print(joint[:5], '\n', joint_processed[:5])
    return joint_processed

sbj_train = ['sbj1', 'sbj2', 'sbj4', 'sbj5', 'sbj9', 'sbj10', 'sbj11', 'sbj12', 'sbj13']
sbj_val = ['sbj15', 'sbj16', 'sbj17']

class SoulartDataset(Dataset):
    def __init__(self, config):
        self.root_dir = config.root_dir
        self.device_num = 'device_01'
        self.T = config.n_frames
        self.n_joints = config.n_joints
        self.n_points = config.n_points
        self.mode = config.mode
        if self.mode == "train":
            self.sbj_modes = sbj_train
        elif self.mode == "val":
            self.sbj_modes = sbj_val
        else:
            raise NotImplementedError()

        self.data = []
        self._prepare_data()

    def __len__(self):
        return len(self.data)
    
    def _prepare_data(self):
        _subjects = sorted(os.listdir(self.root_dir))  # List all subjects
        subjects = [sbj for sbj in _subjects if sbj in self.sbj_modes]
        for subject in subjects:
            subject_path = os.path.join(self.root_dir, subject)
            if os.path.isdir(subject_path):
                motions = sorted(os.listdir(subject_path))
                for motion in motions:
                    sbj_motion_path = os.path.join(subject_path, motion)
                    if os.path.isdir(sbj_motion_path):
                        curr_ply_path = os.path.join(sbj_motion_path, "point_cloud", self.device_num)
                        curr_imu_path = os.path.join(sbj_motion_path, "motion")
                        if os.path.isdir(curr_ply_path) and os.path.isdir(curr_imu_path):
                            frames = sorted(os.listdir(curr_ply_path))
                            imus = sorted(os.listdir(curr_imu_path))
                            frames_time = [float(f"{frame.split('_')[1]}.{frame.split('_')[2].split('.')[0]}") for frame in frames]
                            imus_time = np.array([float(f"{imu.split('_')[1]}.{imu.split('_')[2].split('.')[0]}") for imu in imus])
                            # print(imus_time, frames_time)
                            
                            frame_paths = [os.path.join(curr_ply_path, frame) for frame in frames if frame.endswith('.ply')]
                            imu_paths = [os.path.join(curr_imu_path, imu) for imu in imus if imu.endswith('.csv')]
                            # print(frame_paths, frames_time)
                            # print(imu_paths, imus_time)

                            for j in range(len(frame_paths) - self.T + 1):
                                _data = []
                                for k in range(j, j + self.T):
                                    frame = frame_paths[k]
                                    frame_time = frames_time[k]
                                    frame_imu_diff = np.abs(imus_time - frame_time)
                                    imu_idx = np.argmin(frame_imu_diff)
                                    # print(imu_idx)
                                    _data.append((frame, imu_paths[imu_idx]))
                                self.data.append(_data)


    def __getitem__(self, idx): 
        point_clouds = []
        x_datas = []
        r_datas = []
        seq_cetroid = None
        seq_m = None
        cetroid_fin = None

        for i in range(self.T):
            frame_path, imu_path = self.data[idx][i]

            if i % 5 == 0:
                # only store t, t-5, t-10, t-15, t-20 th frames 
                points = read_point_cloud_ply(frame_path)
                _, cetroid = pc_norm(points)
                points = points - cetroid
                points = random_point_sample(points, self.n_points) # (self.n_points, 3)
                points = points + cetroid
                point_clouds.append(points)
            
            if i == self.T - 2 or i == self.T - 1:
                joints, rotations_axis, rotations_6d = process_csv(imu_path) # J, 3 / J, 3 (axis-angle representation) / J, 6 (6d representation)
                joints = joints - joints[0:1, :] # make the root-relative joint coordinate
                _abs_pose = compute_global_rotation_matrix_from_axis_angle(rotations_axis.view(-1), get_ours_skeleton())
                joints_rel = calc_rel_joint(joints.reshape((-1, 3)).astype(np.float32), _abs_pose, get_ours_skeleton())
        
                x_datas.append(torch.from_numpy(joints_rel))
                r_datas.append(rotations_6d)

            # _t
            if i == self.T - 1:
                cetroid_fin = cetroid
                seq_cetroid = torch.tensor(cetroid, dtype=torch.float32)
                seq_m = torch.tensor(1.0, dtype=torch.float32)
        
        point_clouds = np.stack(point_clouds, axis=0)  # Shape: (T, self.n_points, 3)
        point_clouds = point_clouds - cetroid_fin[None, None, :]
        # point_clouds = point_clouds / m
        point_clouds = torch.from_numpy(point_clouds).to(torch.float32) # (T, self.n_points, 3)


        xs = torch.stack(x_datas, axis=0).to(torch.float32) # T, J, 3
        # note that we already made root-relative coordinate in L166
        rs = torch.stack(r_datas, axis=0).to(torch.float32) # T, J, 6
        
        x_t = torch.cat(
            (
                xs[-1],
                rs[-1]
            ),
            axis=-1
        )
        x_t[0:1, 3:] = torch.tensor([1., 0., 0., 0., 1., 0.]) 

        x_t_prev = torch.cat(
            (
                xs[-2],
                rs[-2]
            ),
            axis=-1
        )
        x_t_prev[0:1, 3:] = torch.tensor([1., 0., 0., 0., 1., 0.]) 

        g_t = torch.cat(
            (
                torch.zeros(size=(3,)),
                rs[-1][0]
            ),
            axis=-1
        ) # 9
        g_t_prev = torch.cat(
            (
                torch.zeros(size=(3,)),
                rs[-2][0]
            ),
            axis=-1
        ) # 9

        Item = {
            'pc': point_clouds,
            'x_t': x_t,
            'x_t_prev': x_t_prev,
            'g_t': g_t,
            'g_t_prev': g_t_prev,
            'cetroid': seq_cetroid,
            'scale': seq_m
        }

        return Item 
