from pathlib import Path
import torch
import datetime

class Config:
    def __init__(self, experiment=None, model=None, project_root_dir=None,
                 joints_set=None, loss_type=None, mkdir=True, normalize=False,
                 r6d=False, device=None, use_joint_loss=False, use_glb_rot_loss=False,
                 use_acc_recon_loss=False, pred_joints_set=None, pred_last_frame=False,
                 use_vposer_loss=False, use_vel_loss=False):
        self.experiment = experiment
        self.model = model
        self.root_dir = Path(project_root_dir).absolute()
        self.joints_set = joints_set
        self.pred_joints_set = [*range(24)] if pred_joints_set == None else pred_joints_set

        self.mkdir = mkdir
        self.normalize = normalize
        self.r6d = r6d
        self.use_joint_loss = use_joint_loss
        self.use_glb_rot_loss = use_glb_rot_loss 
        self.use_acc_recon_loss = use_acc_recon_loss
        self.pred_last_frame = pred_last_frame
        self.use_vposer_loss = use_vposer_loss
        self.use_vel_loss = use_vel_loss

        if device != None:
            if 'cpu' in device:
                self.device = torch.device(f'cpu')
            else:
                self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.build_paths()

        self.loss_type = loss_type
    
    def build_paths(self):
        self.smpl_model_path = self.root_dir / "src/imuposer/smpl/model.pkl"
        self.og_smpl_model_path = self.root_dir / "src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
        
        self.raw_dip_path = self.root_dir / "data/raw/DIP_IMU"
        self.raw_amass_path = self.root_dir / "data/raw/AMASS"

        self.processed_imu_poser = self.root_dir / "data/processed_imuposer"
        self.processed_imu_poser_25fps = self.root_dir / "data/processed_imuposer_25fps"

        self.vposer_ckpt_path = self.root_dir / "extern/vposer_v2_05"

        if self.mkdir:
            if self.experiment != None:
                datestring = datetime.datetime.now().strftime("%m%d%Y-%H%M%S")
                self.checkpoint_path = self.root_dir / f"checkpoints/{self.experiment}-{datestring}"
                self.checkpoint_path.mkdir(exist_ok=True, parents=True)
            else:
                print("No experiment name give, can't create dir")

    max_sample_len = 300
    acc_scale = 30
    train_pct = 0.9
    batch_size = 256
    torch_seed = 0

# DIP order
# 
# 0 head,
# 1 spine2,
# 2 belly,
# 3 lchest,
# 4 rchest,
# 5 lshoulder,
# 6 rshoulder,
# 7 lelbow,
# 8 relbow,
# 9 lhip,
# 10 rhip,
# 11 lknee,
# 12 rknee,
# 13 lwrist,
# 14 rwrist,
# 15 lankle,
# 16 rankle

# head_rlwrist_rlpocket
# 0 (head)
# 14 (right wrist)
# 13 (left wrist)
# 10 (right hip)
# 9 (left hip)

imuName2idx = {
    "lw": 0,
    "rw": 1,
    "lp": 2,
    "rp": 3,
    "h": 4
}

amass_combos = {
    'global': [0, 1, 2, 3, 4],
    'lw_rw_h': [0, 1, 4],
    'rw_lp_rp': [1, 2, 3],
    'lw_rw_rp': [0, 1, 3],
    'lw_rp_h': [0, 3, 4],
    'rw_rp_h': [1, 3, 4],
    'lw_lp_rp': [0, 2, 3],
    'lw_rw_lp': [0, 1, 2],
    'lw_lp_h': [0, 2, 4],
    'rw_lp_h': [1, 2, 4],
    'lw_rw': [0, 1],
    'lw_lp': [0, 2],
    'lw_rp': [0, 3],
    'lw_h': [0, 4],
    'rw_lp': [1, 2],
    'rw_rp': [1, 3],
    'rw_h': [1, 4],
    'lp_rp': [2, 3],
    'lp_h': [2, 4],
    'rp_h': [3, 4],
    'lw': [0],
    'rw': [1],
    'lp': [2],
    'rp': [3],
    'h': [4]
 }


pred_joints_set = {
    "legs": [0, 1, 2, 4, 5, 7, 8, 10, 11], 
    "upper_body": [0, 3, 6, 9, 13, 14, 16, 17, 18, 19, 20, 21], 
    "head": [0, 12, 15],
}

# Add more here if you want
amass_datasets = ['ACCAD', 'BioMotionLab_NTroje', 'BMLhandball', 'BMLmovi', 'CMU',
                  'DanceDB', 'DFaust_67', 'EKUT', 'Eyes_Japan_Dataset', 'HUMAN4D',
                  'HumanEva', 'KIT', 'MPI_HDM05', 'MPI_Limits', 'MPI_mosh', 'SFU',
                  'SSM_synced', 'TCD_handMocap', 'TotalCapture', 'Transitions_mocap']

leaf_joints = [20, 21, 7, 8, 12]


limb2vertexkeys = {
    "LLeg": ["leftLeg", "leftToeBase", "leftFoot", "leftUpLeg"],
    "RLeg": ["rightUpLeg", "rightFoot", "rightLeg", "rightToeBase"],
    "LArm": ["leftArm", "leftHandIndex1", "leftForeArm", "leftHand", "leftShoulder"], 
    "RArm": ["rightArm", "rightHandIndex1", "rightForeArm", "rightHand", "rightShoulder"], 
    "Head": ["head", "neck"], 
    "Torso": ["spine1", "spine2", "spine", "hips"]
}

end_effector2vertexkeys = {
    "LFoot": ["leftFoot"],
    "RFoot": ["rightFoot"],
    "LHand": ["leftHand"], 
    "RHand": ["rightHand"],
    "withoutEndEffectors": ["leftLeg", "leftToeBase", "leftUpLeg", "rightUpLeg", "rightLeg", "rightToeBase",
                            "leftArm", "leftHandIndex1", "leftForeArm", "leftShoulder", "rightArm", "rightHandIndex1", 
                            "rightForeArm", "rightShoulder", "spine1", "spine2", "spine", "hips"]
}

limb2joints = {
    "LLeg": [1, 4, 7, 10],
    "RLeg": [2, 5, 8, 11],
    "LArm": [16, 18, 20, 22],
    "RArm": [17, 19, 21, 23],
    "Head": [15, 12],
    "Torso": [3, 6, 9, 13, 14]
}

end_effector2joints = {
    "LFoot": [7],
    "RFoot": [8],
    "LHand": [20],
    "RHand": [21],
    "withoutEndEffectors": [1, 4, 10,
                            2, 5, 11,
                            16, 18, 22,
                            17, 19, 23,
                            3, 6, 9, 13, 14]
}
