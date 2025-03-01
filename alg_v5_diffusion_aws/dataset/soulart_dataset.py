import torch
import os
import numpy as np
import pandas as pd
from plyfile import PlyData
from torch.utils.data import Dataset

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
    
    joints = torch.from_numpy(joints)
    rotations_6d = matrix_to_rotation_6d(euler_to_matrix(rotations))


    return joints, rotations_6d


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

def farthest_point_sample(xyz, npoint):
    ndataset = xyz.shape[0]
    if ndataset<npoint:
        repeat_n = int(npoint/ndataset)
        xyz = np.tile(xyz,(repeat_n,1))
        xyz = np.append(xyz,xyz[:npoint%ndataset],axis=0)
        return xyz
    centroids = np.zeros(npoint)
    distance = np.ones(ndataset) * 1e10
    farthest =  np.random.randint(0, ndataset)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[int(farthest)]
        dist = np.sum((xyz - centroid) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return xyz[np.int32(centroids)]

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

            points = read_point_cloud_ply(frame_path)
            _, cetroid = pc_norm(points)
            points = points - cetroid
            points = farthest_point_sample(points, self.n_points) # (self.n_points, 3)
            points = points + cetroid
            point_clouds.append(points)
            
            joints, rotations = process_csv(imu_path) # J, 3 / J, 6
            joints = joints - joints[0:1, :] # make teh root-relative joint coordinate
            x_datas.append(joints)
            r_datas.append(rotations)

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
        x = xs[-1]
        r = rs[-1]
        abs_x = torch.zeros(size=(3,)).to(torch.float32)

        Item = {
            'pts': point_clouds,
            'xs': xs,
            'rs': rs,
            'x': x,
            'r': r,
            'abs_x': abs_x,
            'cetroid': seq_cetroid,
            'scale': seq_m
        }

        return Item 

if __name__ == "__main__":
    ds = OurDataset(root_dir = "./data_pipeline/")
    print(len(ds))
    # process_csv("./data_pipeline/sbj1/01_t_pose_01/motion/frame_113547_740222.csv")
    item = ds[0]
    print(item['pts'].shape)
    print(item['xs'].shape)
    print(item['x'].shape)
    print(item['rs'])
    print(item['r'])
    # print(item['x'])
    # print(len(ds))
    
    import open3d as o3d
    point_cloud = o3d.geometry.PointCloud()
    
    pc = item['pts']
    x = item['x']
    # point_cloud.points = o3d.utility.Vector3dVector(torch.from_/numpy(seq_j[-1]).view(-1, 3)) # j_vis
    # point_cloud.points = o3d.utility.Vector3dVector(abs_joint) # j_vis
    # point_cloud.points = o3d.utility.Vector3dVector(pc[-1]) 
    point_cloud.points = o3d.utility.Vector3dVector(x) 
    # Save to .ply file
    o3d.io.write_point_cloud('x.ply', point_cloud)
