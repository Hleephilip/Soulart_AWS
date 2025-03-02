import torch
import numpy as np
import os
import copy
import sys
import yaml
from plyfile import PlyData
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from easydict import EasyDict as edict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.smpl_utils import get_smpl_skeleton, rotation_6d_to_matrix

'''
Auxiliary functions
'''

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

def read_point_cloud_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    ply_data = PlyData.read(filename)['vertex'].data
    points = np.array([[x, y, z] for x, y, z in ply_data])
    return points

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

'''
Dataset class
'''

class lidarhuman26MDataset(Dataset):
    def __init__(self, cfgs):
        self.root_dir = cfgs.root_dir
        self.n_joints = cfgs.n_joints
        self.n_points = cfgs.n_points
        self.mode = cfgs.mode
        self.T = cfgs.n_frames
        self.coord_type = cfgs.joint_coord_type # P or PST

        self.data = []
        self.joint_pos_abs = {} # includes every joint abs xyz coordinates
        self.joint_pos_rel = {} # includes every joint rel xyz coordinates
        self.joint_rot = {}
        self.root_rot = {} # includes axis angle rotation vector [6d] of root joint

        self._prepare_data()

    def __len__(self):
        return len(self.data) 

    def _prepare_data(self):
        """Prepare a list of all valid sequences of T frames."""
        subjects = sorted(os.listdir(os.path.join(self.root_dir, 'labels/3d/segment', self.mode)))  # List all subjects
        for subject in subjects:
            # point cloud
            subject_path = os.path.join(self.root_dir, 'labels/3d/segment', self.mode, subject)
            if os.path.isdir(subject_path):
                frames = sorted(os.listdir(subject_path))
                frame_paths = [os.path.join(subject_path, frame) for frame in frames if frame.endswith('.ply')]
                # Collect sequences of T consecutive frames
                for i in range(len(frame_paths) - self.T + 1):
                    self.data.append((i, subject, frame_paths[i:i + self.T]))

            # joint coords (including root coords)
            self.joint_pos_abs[subject] = torch.load(os.path.join(self.root_dir, f'processed/joint_world_{self.coord_type}', \
                self.mode, f'abs_{subject}.pt'), map_location='cpu') # [total_frames, n_joint, 3], default(n_joint) = 24 (in SMPL setting)
            
            # joint coords (including root coords)
            self.joint_pos_rel[subject] = torch.load(os.path.join(self.root_dir, f'processed/joint_world_{self.coord_type}', \
                self.mode, f'rel_{subject}.pt'), map_location='cpu') # [total_frames, n_joint, 3], default(n_joint) = 24 (in SMPL setting)

            # joint rotation
            self.joint_rot[subject] = torch.load(os.path.join(self.root_dir, f'processed/rotation_6d', \
                self.mode, f'joint_6d_{subject}.pt'), map_location='cpu') # [total_frames, n_joint, 6], default(n_joint) = 24 (in SMPL setting)
            
            # root rotation
            self.root_rot[subject] = torch.load(os.path.join(self.root_dir, f'processed/rotation_6d', \
                self.mode, f'root_6d_{subject}.pt'), map_location='cpu') # [total_frames, 1, 6] (in SMPL setting)

        return 

    def __getitem__(self, idx):
        """Return a sample consisting of T consecutive frames."""
        curr_frame, sbj, frame_paths = self.data[idx]
        point_clouds = []

        # joint_pos_abs = self.joint_pos_abs[sbj][curr_frame + self.T - 1] # (n_joint, 3)
        # joint_pos_rel = self.joint_pos_rel[sbj][curr_frame + self.T - 1] # (n_joint, 3)
        # joint_rot = self.joint_rot[sbj][curr_frame + self.T - 1] # (n_joint, 6)
        # root_rot = self.root_rot[sbj][curr_frame + self.T - 1] # (1, 6)

        x_t = torch.cat(
            (
                self.joint_pos_rel[sbj][curr_frame + self.T - 1],
                self.joint_rot[sbj][curr_frame + self.T - 1]
            ),
            axis=-1
        ) # n_joint, 9
        x_t[0:1, 3:] = torch.tensor([1., 0., 0., 0., 1., 0.]) # since x_t denotes relative imu
           
        x_t_prev = torch.cat(
            (
                self.joint_pos_rel[sbj][curr_frame + self.T - 2],
                self.joint_rot[sbj][curr_frame + self.T - 2]
            ),
            axis=-1
        ) # n_joint, 9
        x_t_prev[0:1, 3:] = torch.tensor([1., 0., 0., 0., 1., 0.]) # since x_t_prev denotes relative imu

        joint_vis = self.joint_pos_abs[sbj][curr_frame + self.T - 1] # J, 3
        g_t = torch.cat(
            (
                self.joint_pos_abs[sbj][curr_frame + self.T - 1][0],
                self.root_rot[sbj][curr_frame + self.T - 1][0]
            ),
            axis=-1
        ) # 9
        g_t_prev = torch.cat(
            (
                self.joint_pos_abs[sbj][curr_frame + self.T - 2][0],
                self.root_rot[sbj][curr_frame + self.T - 2][0]
            ),
            axis=-1
        ) # 9

        seq_cetroid = None
        seq_m = None
        cetroid_fin = None

        for i, t in enumerate(range(curr_frame, curr_frame + self.T)):
            '''
            if i == self.T - 2:
                path = frame_paths[t - curr_frame]
                # Read the point cloud file
                points = read_point_cloud_ply(path)  # Extract point coordinates (N x 3)
                points, cetroid = pc_norm(points)
                g_t_prev[:3] = g_t_prev[:3] - torch.tensor(cetroid, dtype=torch.float32) #/ seq_m
                # x_t_prev[:, :3] = x_t_prev[:, :3] / seq_m # normalize the absolute 3d joint 
            '''

            if i % 5 != 0: continue
            # only store t, t-5, t-10, t-15, t-20 th frames 

            path = frame_paths[t - curr_frame]
            # Read the point cloud file
            points = read_point_cloud_ply(path)  # Extract point coordinates (N x 3)

            _, cetroid = pc_norm(points)
            points = points - cetroid
            points = random_point_sample(points, self.n_points) # (self.n_points, 3)
            points = points + cetroid
            # use random sample instead of FPS since movin utilizes random sampling
            point_clouds.append(points)
            # _t
            if i == self.T - 1:
                cetroid_fin = cetroid
                seq_cetroid = torch.tensor(cetroid, dtype=torch.float32)
                seq_m = torch.tensor(1.0, dtype=torch.float32)
                # seq_m = torch.tensor(m, dtype=torch.float32)

    
        g_t_prev[:3] = g_t_prev[:3] - seq_cetroid # / seq_m
        g_t[:3] = g_t[:3] - seq_cetroid # / seq_m
        joint_vis = joint_vis - seq_cetroid # / seq_m
        # x_t_prev[:, :3] = x_t_prev[:, :3] / seq_m # 
        # x_t[:, :3] = x_t[:, :3] / seq_m # normalize the absolute 3d joint 

        point_clouds = np.stack(point_clouds, axis=0)  # Shape: (T, self.n_points, 3)
        point_clouds = point_clouds - cetroid_fin[None, None, :]
        # point_clouds = point_clouds / m
        point_clouds = torch.from_numpy(point_clouds).to(torch.float32) # (T, self.n_points, 3)

        Item = {
            'pc': point_clouds,
            'x_t': x_t,
            'x_t_prev': x_t_prev,
            'g_t': g_t,
            'g_t_prev': g_t_prev,
            'cetroid': seq_cetroid,
            'scale': seq_m,
            'joint_vis': joint_vis
        }
        # Note that transformation of input data is implemented in training loop.
        return Item

def main():
    cfgs = load_config("../cfgs/lidarhuman26M.yaml")
    train_dataset = lidarhuman26MDataset(cfgs.train_dataset)

    item = train_dataset[0]
    print(item.keys())
    pc = item['pc']
    print(pc.shape)
    x_t = item['x_t']
    print(x_t.shape)
    g_t = item['g_t']
    print(g_t.shape)
    cetroid = item['cetroid']
    print(cetroid.shape)
    scale = item['scale']
    print(scale.shape)
    joint_vis = item['joint_vis']

    print(pc.dtype, x_t.dtype, g_t.dtype, cetroid.dtype, scale.dtype)
    import open3d as o3d
    
    point_cloud = o3d.geometry.PointCloud()
    rel_joint = x_t[:, :3]
    # abs_joint = calc_abs_joint(rel_joint.unsqueeze(0), torch.cat((g_t[3:].unsqueeze(0), x_t[1:, 3:]), axis=0).unsqueeze(0), get_smpl_skeleton())[0] + g_t[:3] 
    abs_joint = calc_abs_joint(rel_joint.unsqueeze(0), x_t[:, 3:].unsqueeze(0), get_smpl_skeleton())[0] + g_t[:3] 
    # final version
    # change axis=1 in batched situation
    # print(abs_joint)

    # point_cloud.points = o3d.utility.Vector3dVector(torch.from_/numpy(seq_j[-1]).view(-1, 3)) # j_vis
    point_cloud.points = o3d.utility.Vector3dVector(np.array(joint_vis.detach().clone().cpu())) # j_vis
    # point_cloud.points = o3d.utility.Vector3dVector(abs_joint) 
    # point_cloud.points = o3d.utility.Vector3dVector(pc[-1]) 
    # Save to .ply file
    o3d.io.write_point_cloud('joint_vis.ply', point_cloud)
    # all float32
if __name__ == "__main__":
    main()