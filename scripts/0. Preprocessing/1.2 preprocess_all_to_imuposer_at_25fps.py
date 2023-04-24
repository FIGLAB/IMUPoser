r"""
    Resample 60fps datasets to 25fps
    Smoothen with an average filter
"""

# +
import torch
from tqdm import tqdm

from imuposer.config import Config
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math
# -

config = Config(project_root_dir="../../")

# +
target_fps = 25
# hop = 60 // target_fps

def smooth_avg(acc=None, s=3):
    nan_tensor = (torch.zeros((s // 2, acc.shape[1], acc.shape[2])) * torch.nan)
    acc = torch.cat((nan_tensor, acc, nan_tensor))
    tensors = []
    for i in range(s):
        L = acc.shape[0]
        tensors.append(acc[i:L-(s-i-1)])

    smoothed = torch.stack(tensors).nanmean(dim=0)
    return smoothed

def _resample(tensor, target_fps):
    r"""
        Resample to the target fps, assumes 60fps input
    """
    indices = torch.arange(0, tensor.shape[0], 60/target_fps)

    start_indices = torch.floor(indices).long()
    end_indices = torch.ceil(indices).long()
    end_indices[end_indices >= tensor.shape[0]] = tensor.shape[0] - 1 # handling edge cases

    start = tensor[start_indices]
    end = tensor[end_indices]
    
    floats = indices - start_indices
    for shape_index in range(len(tensor.shape) - 1):
        floats = floats.unsqueeze(1)
    weights = torch.ones_like(start) * floats
    torch_lerped = torch.lerp(start, end, weights)
    return torch_lerped


# -
path_to_save = config.processed_imu_poser_25fps
path_to_save.mkdir(exist_ok=True, parents=True)

# 11 frames at 60 fps = 11*25/60
11*25/60

# process AMASS first
for fpath in (config.processed_like_transpose_path / "AMASS").iterdir():
    # resample to 25 fps
    joint = [_resample(x, target_fps) for x in torch.load(fpath / "joint.pt")]
    pose = [math.axis_angle_to_rotation_matrix(_resample(x, target_fps).contiguous()).view(-1, 24, 3, 3) for x in torch.load(fpath / "pose.pt")]
    shape = torch.load(fpath / "shape.pt")
    tran = [_resample(x, target_fps) for x in torch.load(fpath / "tran.pt")]
    
    # average filter
    vacc = [smooth_avg(_resample(x, target_fps), s=5) for x in torch.load(fpath / "vacc.pt")]
    vrot = [_resample(x, target_fps) for x in torch.load(fpath / "vrot.pt")]
    
    # save the data
    fdata = {
        "joint": joint,
        "pose": pose,
        "shape": shape,
        "tran": tran,
        "acc": vacc,
        "ori": vrot
    }
    
    torch.save(fdata, path_to_save / f"{fpath.name}.pt")

# process DIP next
for fpath in (config.processed_like_transpose_path / "DIP_IMU").iterdir():
    # resample to 25 fps
    joint = [_resample(x, target_fps) for x in torch.load(fpath / "joint.pt")]
    pose = [math.axis_angle_to_rotation_matrix(_resample(x, target_fps).contiguous()).view(-1, 24, 3, 3) for x in torch.load(fpath / "pose.pt")]
    shape = torch.load(fpath / "shape.pt")
    tran = [_resample(x, target_fps) for x in torch.load(fpath / "tran.pt")]
    
    # average filter
    acc = [smooth_avg(_resample(x, target_fps), s=5) for x in torch.load(fpath / "accs.pt")]
    rot = [_resample(x, target_fps) for x in torch.load(fpath / "oris.pt")]
    
    # save the data
    fdata = {
        "joint": joint,
        "pose": pose,
        "shape": shape,
        "tran": tran,
        "acc": acc,
        "ori": rot
    }
    
    torch.save(fdata, path_to_save / f"dip_{fpath.name}.pt")


# +
# # process DIP next
# # dict_keys(['acc', 'ori', 'pose', 'tran'])
# dip_data = torch.load(config.processed_like_transpose_path / "DIP_IMU/train.pt")
# acc = [smooth_avg(_resample(x, target_fps), s=5) for x in dip_data["acc"]]
# ori = [_resample(x, target_fps) for x in dip_data["ori"]]
# pose = [math.axis_angle_to_rotation_matrix(_resample(x, target_fps).contiguous()).view(-1, 24, 3, 3)for x in dip_data["pose"]]
# tran = [_resample(x, target_fps) for x in dip_data["tran"]]
# 
# # save the data
# fdata = {
#     "pose": pose,
#     "tran": tran,
#     "acc": acc,
#     "ori": ori
# }
# 
# torch.save(fdata, path_to_save / f"dip_train.pt")

# +
# # dict_keys(['acc', 'ori', 'pose', 'tran'])
# dip_data = torch.load(config.processed_like_transpose_path / "DIP_IMU/test.pt")
# acc = [smooth_avg(_resample(x, target_fps), s=5) for x in dip_data["acc"]]
# ori = [_resample(x, target_fps) for x in dip_data["ori"]]
# pose = [math.axis_angle_to_rotation_matrix(_resample(x, target_fps).contiguous()).view(-1, 24, 3, 3)for x in dip_data["pose"]]
# tran = [_resample(x, target_fps) for x in dip_data["tran"]]
# 
# # save the data
# fdata = {
#     "pose": pose,
#     "tran": tran,
#     "acc": acc,
#     "ori": ori
# }
# 
# torch.save(fdata, path_to_save / f"dip_test.pt")
# -

# pose[0].shape
# 
# dip_data.keys()
# 
# 
# # +
# # joint[0].shape, shape[0].shape, tran[0].shape, vacc[0].shape, vrot[0].shape
# # -
# 
# # ## Sanity checks
# 
# def syn_acc(v):
#     acc = torch.stack([(v[i] + v[i+2] - 2 * v[i+1]) * 3600 for i in range(0, v.shape[0] - 2)])
#     acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
#     return acc
# 
# 
# pose = dip_data["pose"][0].view(-1, 24, 3)
# tran = dip_data["tran"][0]
# 
# dip_data.keys()
# 
# body_model = ParametricModel(config.og_smpl_model_path)
# 
# # +
# vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
# ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])
# 
# shape = torch.ones((1, 10))
# 
# p = math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
# grot, joint, vert = body_model.forward_kinematics(p, shape[0], tran, calc_mesh=True)
# vacc = syn_acc(vert[:, vi_mask])
# vrot = grot[:, ji_mask]
# # -
# 
# vrot.shape
# 
# ori = dip_data["ori"][0]
# 
# import matplotlib.pyplot as plt
# 
# plt.figure(figsize=(30, 10))
# plt.plot(vrot[:, 2, 0, 0])
# plt.plot(ori[:, 2, 0, 0])
# # plt.ylim([-2, 2])
# 
# acc = dip_data["acc"][0]
# 
# 
# 
# acc.shape
# 
# plt.figure(figsize=(30, 10))
# plt.plot(vacc[:, 2, 1])
# plt.plot(acc[:, 2, 1])
# plt.ylim([-2, 2])
# 
# plt.figure(figsize=(30, 10))
# plt.plot(smooth_avg(vacc, 11)[:, 2, 1])
# plt.plot(smooth_avg(acc, 11)[:, 2, 1])
# plt.ylim([-2, 2])
# 
# 
