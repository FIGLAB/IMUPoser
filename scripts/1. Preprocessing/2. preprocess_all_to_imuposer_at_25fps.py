r"""
    Resample 60fps datasets to 25fps
    Smoothen with an average filter
"""

# +
import torch

from imuposer.config import Config
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
for fpath in (config.processed_imu_poser / "AMASS").iterdir():
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
for fpath in (config.processed_imu_poser / "DIP_IMU").iterdir():
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

