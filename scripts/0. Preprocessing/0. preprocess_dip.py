# %load_ext autoreload
# %autoreload 2

import numpy as np
import pickle as pkl

from imuposer.smpl.smpl_np import SMPLModel
from imuposer.config import Config

config = Config(project_root_dir="../../")

smpl = SMPLModel(config.smpl_model_path)
trans = np.zeros(smpl.trans_shape)
beta= np.zeros(smpl.beta_shape)

path_to_raw_dip = config.raw_dip_path

path_to_save = path_to_raw_dip.parent.parent / "processed/DIP_IMU"
path_to_save.mkdir(exist_ok=True, parents=True)

SMPL_NR_JOINTS = 24
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

def convert_to_rodrigues(pose_rvec):
    pose_rodrigues = []
    for i in range(len(pose_rvec)):
        smpl.set_params(beta=beta, pose=pose_rvec[i], trans=trans)
        pose_rodrigues.append(smpl.R)
    return np.array(pose_rodrigues)

def smpl_rot_to_global(smpl_rotations_local):
    in_shape = smpl_rotations_local.shape
    do_reshape = in_shape[-1] != 3
    if do_reshape:
        assert in_shape[-1] == 216
        rots = np.reshape(smpl_rotations_local, in_shape[:-1] + (SMPL_NR_JOINTS, 3, 3))
    else:
        rots = smpl_rotations_local

    out = np.zeros_like(rots)
    dof = rots.shape[-3]
    for j in range(dof):
        if SMPL_PARENTS[j] < 0:
            out[..., j, :, :] = rots[..., j, :, :]
        else:
            parent_rot = out[..., SMPL_PARENTS[j], :, :]
            local_rot = rots[..., j, :, :]
            out[..., j, :, :] = np.matmul(parent_rot, local_rot)

    if do_reshape:
        out = np.reshape(out, in_shape)

    return out

acc_scale = 30 

for folder in path_to_raw_dip.iterdir():
    pid = folder.name
    for file in folder.iterdir():
        print(file.name)
        acc_data = []
        ori_data = []
        gt_data = []
        with open(file, 'rb') as f:
            data = pkl.load(f, encoding='latin1')
            acc_data.append(np.nan_to_num(data['imu_acc']) / acc_scale)
            ori_data.append(np.nan_to_num(data['imu_ori']))
            pose_dat = np.nan_to_num(data['gt'])
            pose_dat_R = convert_to_rodrigues(pose_dat)
            pose_dat_R = pose_dat_R.reshape(pose_dat_R.shape[0],216)
            pose_dat_ = pose_dat_R
            pose_dat_R = smpl_rot_to_global(pose_dat_R)
            gt_data.append(pose_dat_R)
        
        save_data = {
            "acc": acc_data,
            "ori": ori_data,
            "global_pose": gt_data,
            "pose": pose_dat_
        }
        
        with open(path_to_save / f"{pid}_{file.name}", "wb") as f:
            pkl.dump(save_data, f)
