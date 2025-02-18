import torch
import argparse
import pickle5 as pickle
import numpy as np
import os
import sys
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R
from easydict import EasyDict as edict

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.smpl_utils import axis_angle_to_matrix, matrix_to_rotation_6d, get_smpl_skeleton, rotation_6d_to_matrix

'''
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
'''
def load_config(path):
    with open(path) as fin:
        config = edict(yaml.safe_load(fin))

    return config


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


def calc_rel_joint(joint, rotations, joint_tree):
    joint_processed = np.zeros(joint.shape)
    # joint_processed[0] = joint[0]
    for i in range(1, 24):
        parent = joint_tree[i-1][0]
        diff = torch.from_numpy(joint[i] - joint[parent])
        joint_processed[i, :] = np.array(torch.einsum("ij, j -> i", rotations[parent].transpose(-1, -2), diff))

    # print(joint[:5], '\n', joint_processed[:5])
    return joint_processed

class LIPDDataset(torch.utils.data.Dataset):
    def __init__(self, cfgs):
        self.root_dir = cfgs.root_dir
        self.n_joints = cfgs.n_joints
        self.num_points = cfgs.n_points
        self.dataset = []
        self.use_rel_coodinate = True # default
        self.use_6d_rotation = True # default
        m = cfgs.mode
        self.T = cfgs.n_frames
        data_info_path = self.root_dir #'LIPD_test.pkl' or LIPD_train.pkl

        file = open(data_info_path,'rb')
        datas = pickle.load(file)
        file.close()
        old_motion_id = datas[0]['seq_path']

        seq = []
        j=0
        if self.T == 1:
            self.dataset = datas
        else:
            while True:
                if j >=len(datas):
                    break
                motion_id = datas[j]['seq_path']
                if motion_id==old_motion_id:
                    seq.append(datas[j])
                    j+=1
                else:
                    old_motion_id = motion_id
                    seq=[datas[j]]
                    j+=1
                if len(seq) == self.T:
                    self.dataset.append(seq)
                    seq=[]
    
    def __getitem__(self, index):
        example_seq = self.dataset[index]
        seq_pc = []
        seq_gt = []
        seq_j = []
        seq_j_rel = []
        seq_r = []
        seq_imu_ori = []
        seq_imu_acc = []
        seq_trans = []
        seq_cetroid = None
        seq_m = None
        for i, example in enumerate(example_seq):
            pc_data = example['pc']
            pc_data = os.path.join('/home/coder/project/DataA/LIPD', pc_data)
            if type(pc_data)==str:
                pc_data = np.fromfile(pc_data,dtype=np.float32).reshape(-1,3)
            if len(pc_data)==0:
                pc_data = np.array([[0,0,0]])
            cetroid = pc_data.mean(0)
            pc_data = pc_data - pc_data.mean(0)
            pc_data = random_point_sample(pc_data, self.num_points) # (self.n_points, 3)
            # use random sample instead of FPS since movin utilizes random sampling
            # pc_data = farthest_point_sample(pc_data,self.num_points)

            if i % 5 == 0:
                seq_pc.append(pc_data)
                # only store t, t-5, t-10, t-15, t-20 th frames 
            gt_r = example['gt_r'] # contains 6d representation of root rotation
            gt_j = example['gt_joint'].reshape(-1) # [72] (Joint Coords.)
            gt = example['gt'] # [72] (Pose; rotation)
            
            if self.use_rel_coodinate:
                _abs_pose = compute_global_rotation_matrix_from_axis_angle(torch.from_numpy(gt), get_smpl_skeleton())
                joint_rel = calc_rel_joint(gt_j.reshape((-1, 3)).astype(np.float32), _abs_pose, get_smpl_skeleton()) # 
            
            if self.use_6d_rotation:
                gt_pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(gt).view(-1,3))).view(-1,6) # 24, 6
                seq_gt.append(gt_pose_6d)
            else:
                seq_gt.append(gt)
            seq_r.append(gt_r) # only root rotation
            seq_j.append(gt_j)
            seq_j_rel.append(joint_rel) 
            seq_imu_ori.append(example['imu_ori'].reshape([-1,9]))
            seq_imu_acc.append(example['imu_acc'].reshape([-1,3]))
            # seq_trans.append(example['trans'])

            if i == self.T - 1:
                seq_cetroid = torch.tensor(cetroid, dtype=torch.float32)
                seq_m = torch.tensor(1.0, dtype=torch.float32)

        x_t = torch.cat(
            (
                torch.from_numpy(seq_j_rel[-1]),
                seq_gt[-1]
            ),
            axis=-1
        ).to(torch.float32)
        x_t[0:1, 3:] = torch.tensor([1., 0., 0., 0., 1., 0.]) # since x_t denotes relative imu
           
        
        x_t_prev = torch.cat(
            (
                torch.from_numpy(seq_j_rel[-2]),
                seq_gt[-2]
            ),
            axis=-1
        ).to(torch.float32)
        x_t_prev[0:1, 3:] = torch.tensor([1., 0., 0., 0., 1., 0.]) # since x_t_prev denotes relative imu


        g_t = torch.cat(
            (
                torch.zeros(size=(3,)),
                torch.from_numpy(seq_r[-1])
            ),
            axis=-1
        ).to(torch.float32)

        g_t_prev = torch.cat(
            (
                torch.zeros(size=(3,)),
                torch.from_numpy(seq_r[-2])
            ),
            axis=-1
        ).to(torch.float32)

        point_clouds = torch.from_numpy(np.array(seq_pc)).to(torch.float32)
        Item = {
            'seq_j': seq_j,
            'pc': point_clouds,
            'x_t': x_t,
            'x_t_prev': x_t_prev,
            'g_t': g_t,
            'g_t_prev': g_t_prev,
            'cetroid': seq_cetroid,
            'scale': seq_m
        }

        # Item = {
        #     'data': np.array(seq_pc),
        #     'gt_j': np.array(seq_j),
        #     'gt_r': np.array(seq_r),
        #     'gt_pose' : np.array(seq_gt),
        #     'id':example['seq_path'],
        #     'imu_ori':np.array(seq_imu_ori),
        #     'imu_acc':np.array(seq_imu_acc),
        #     # 'trans':np.array(seq_trans)
        # }
        return Item

    def __len__(self):
        return len(self.dataset)
    
'''
class Dataset_est(torch.utils.data.Dataset):
    def __init__(self,args,module,pc=True):
        self.dataset = []
        self.pc = pc
        root_dataset_path = args.root_dataset_path
        dis_result = args.dis_result
        m = module
        if m == 'eDIP':
            data_info_path = root_dataset_path+'DIP_test.pkl'
            self.pc_data = np.load(os.path.join(dis_result,"DIP","dis.pkl"),allow_pickle=True)
        elif m=='eTC':
            data_info_path = root_dataset_path+'TC_test.pkl'
            self.pc_data = np.load(os.path.join(dis_result,"TC","dis.pkl"),allow_pickle=True)
        elif m =='eLIPD':
            data_info_path = root_dataset_path+'LIPD_test.pkl'
            self.pc_data = np.load(os.path.join(dis_result,"LIPD","dis.pkl"),allow_pickle=True)
        elif m == 'eLH':
            data_info_path = root_dataset_path+'Test_lidarhuman.pkl'
            self.pc_data = np.load(os.path.join(dis_result,"LH","dis.pkl"),allow_pickle=True)
        else:
            data_info_path =root_dataset_path+'LIPD_train.pkl'
            self.pc_data = np.load(os.path.join(dis_result,"LIPD_Train","dis.pkl"),allow_pickle=True)

        self.pc_data = np.concatenate(self.pc_data).reshape(-1,32,78)
        self.num_points = args.num_points
        T = args.frames

        file = open(data_info_path,'rb')
        datas = pickle.load(file)
        file.close()
        old_motion_id = datas[0]['seq_path']  

        seq = []
        j=0
        if T == 1:
            self.dataset = [datas]
        else:
            while True: 
                if j >=len(datas):
                    break
                motion_id = datas[j]['seq_path']
                if motion_id == old_motion_id:
                    seq.append(datas[j])
                    j+=1
                else:
                    old_motion_id = motion_id
                    seq=[datas[j]]
                    j+=1
                if len(seq) == T:
                    self.dataset.append(seq)
                    seq=[]

    def __getitem__(self, index):
        example_seq = self.dataset[index]
        pc_data = np.array(self.pc_data[index]) # 32 x 78
        seq_pc = []
        seq_gt = []
        seq_j = []
        seq_imu_ori = []
        seq_imu_acc = []
        for example in example_seq:
            if self.pc:
                pc_path = example['pc']
                if len(pc_data)==0:
                    print(pc_path)
                pc_data = pc_data - pc_data.mean(0)
                pc_data = farthest_point_sample(pc_data,self.num_points)
                seq_pc.append(pc_data)
        
            gt = example['gt']
            gt_j = example['gt_joint']
            seq_imu_ori.append(example['imu_ori'].reshape([-1,9]))
            seq_imu_acc.append(example['imu_acc'].reshape([-1,3]))
            seq_j.append(gt_j)
            seq_gt.append(gt)

        Item = {
            'data':np.array(pc_data),
            'gt_pose' : np.array(seq_gt),
            'gt_j': np.array(seq_j),
            'id':example['seq_path'],
            'imu_ori':np.array(seq_imu_ori),
            'imu_acc':np.array(seq_imu_acc),
        }
        if self.pc:
            Item['data'] = np.array(seq_pc)
        return Item
    
    def __len__(self):
        return len(self.pc_data)

class Dataset_ptc(torch.utils.data.Dataset):
    def __init__(self,args,m):
        self.dataset = []
        root_dataset_path = args.root_dataset_path
        self.m = m
        est_result = args.est_result
        if m == 'e':
            data_info_path = os.path.join(root_dataset_path,'LIPD_test.pkl')
            self.pose_path = est_result
            self.pose_data = pickle.load(open(self.pose_path,'rb'))
        else:
            data_info_path = root_dataset_path+'Trans_train.pkl'
  
        self.num_points = args.num_points
        T = args.frames

        file = open(data_info_path,'rb')
        datas = pickle.load(file)
        file.close()
        old_motion_id = datas[0]['seq_path']
        seq = []
        j=0
        if T == 1:
            self.dataset = [datas]
        else:
            while True:
                if j >=len(datas):
                    break
                motion_id = datas[j]['seq_path']
                if motion_id==old_motion_id:
                    seq.append(datas[j])
                    j+=1
                else:
                    old_motion_id = motion_id
                    seq=[datas[j]]
                    j+=1
                if len(seq) == T:
                    self.dataset.append(seq)
                    seq=[]
    
    def __getitem__(self, index):
        example_seq = self.dataset[index]
        seq_pc = []
        seq_gtp = []
        seq_gtj = []
        seq_pose = []
        seq_j = []
        seq_norm = []
        seq_trans = []
        for example in example_seq:
            pc_data = example['pc']
            if type(pc_data)==str:
                pc_data = np.fromfile(pc_data,dtype=np.float32).reshape(-1,3)
            if len(pc_data)==0:
                pc_data=np.array([[0,0,0]])
            norm_trans = pc_data.mean(0)
            pc_data = pc_data-norm_trans
            pc_data = farthest_point_sample(pc_data,self.num_points)
            seq_pc.append(pc_data)

            gt = example['gt']
            gt_j = example['gt_joint']
            seq_gtj.append(gt_j)
            seq_gtp.append(gt)
            if self.m == 'e':
                seq_pose = self.pose_data['pose'][index]
                seq_j = self.pose_data['joint'][index]
            else:
                seq_trans.append(example['trans'])
            seq_norm.append(norm_trans)
            
        Item = {
            'pose' : np.array(seq_pose),
            'joint' : np.array(seq_j),
            'gt_pose': np.array(seq_gtp),
            'gt_joint': np.array(seq_gtj),
            'id':example['seq_path'],
            'norm':np.array(seq_norm),
            'data':np.array(seq_pc),
        }
        if self.m != 'e':
            Item['trans']:np.array(seq_trans)
        return Item
    
    def __len__(self):
        return len(self.dataset)
'''


def main():
    
    cfgs = load_config("../cfgs/LIPD.yaml")
    train_dataset = LIPDDataset(cfgs.train_dataset)
    # print(len(train_dataset)) # 2467
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
    seq_j = item['seq_j']
    print(pc.dtype, x_t.dtype, g_t.dtype, cetroid.dtype, scale.dtype)
    # all float32

    point_cloud = o3d.geometry.PointCloud()
    rel_joint = x_t[:, :3]
    abs_joint = calc_abs_joint(rel_joint.unsqueeze(0), x_t[:, 3:].unsqueeze(0), get_smpl_skeleton())[0] + g_t[:3] 
    # point_cloud.points = o3d.utility.Vector3dVector(torch.from_numpy(seq_j[-1]).view(-1, 3)) # j_vis
    point_cloud.points = o3d.utility.Vector3dVector(abs_joint) # j_vis
    # Save to .ply file
    o3d.io.write_point_cloud('pc.ply', point_cloud)

    '''
    # print(item['data'].shape) # pc: T, 256, 3
    pc = item['data']
    # print(pc[0])
    j = item['gt_j']
    print(j.shape)
    print(item['gt_pose'].shape) # T, 72
    j_vis = torch.from_numpy(j[0]).view(-1, 3)
            
    print(j[0])
    # print(item['gt_j'].shape) # T, 72
    gt_pose = torch.from_numpy(item['gt_pose']) 
    T = args.frames
    gt_pose_6d = matrix_to_rotation_6d(axis_angle_to_matrix(gt_pose.view(T,-1,3))).view(T,-1,6)
    # print(item['imu_ori'].shape) # T, 6, 9
    # print(item['imu_acc'].shape) # T, 6, 3
    print(gt_pose_6d[0][0])
    # print(item['gt_r'][0])
    '''

if __name__ == "__main__":
    main()
    