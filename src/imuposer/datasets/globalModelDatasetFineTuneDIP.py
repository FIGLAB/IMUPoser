import torch
from torch.utils.data import Dataset
from imuposer import math
from imuposer.config import Config, amass_combos

class GlobalModelDatasetFineTuneDIP(Dataset):
    def __init__(self, split="train", config:Config=None):
        super().__init__()

        # load the data
        self.train = split
        self.config = config
        self.data = self.load_data()
        
    def load_data(self):
        if self.train == "train":
            data_files = ["dip_train.pt"]
        else:
            data_files = ["dip_test.pt"]

        imu = []
        pose = []

        for fname in data_files:
            fdata = torch.load(self.config.processed_imu_poser_25fps / fname)

            for i in range(len(fdata["acc"])):
                # inputs
                facc = fdata["acc"][i] 
                fori = fdata["ori"][i]

                # load all the data
                glb_acc = facc.view(-1, 6, 3)[:, [0, 1, 2, 3, 4]] / self.config.acc_scale
                glb_ori = fori.view(-1, 6, 3, 3)[:, [0, 1, 2, 3, 4]]

                acc = glb_acc
                ori = glb_ori

                # outputs
                fpose = fdata["pose"][i]
                fpose = fpose.reshape(fpose.shape[0], -1)

                # clip the data
                # 25 is the data sampling rate

                window_length = self.config.max_sample_len * 25 // 60

                for _combo in list(amass_combos):
                    # acc N, 5, 3
                    # ori N, 5, 3, 3

                    _combo_acc = torch.zeros_like(acc)
                    _combo_ori = torch.zeros((3, 3)).repeat(ori.shape[0], 5, 1, 1)

                    _combo_acc[:, amass_combos[_combo]] = acc[:, amass_combos[_combo]]
                    _combo_ori[:, amass_combos[_combo]] = ori[:, amass_combos[_combo]]

                    imu_inputs = torch.cat([_combo_acc.flatten(1), _combo_ori.flatten(1)], dim=1)

                    imu.extend(torch.split(imu_inputs, window_length))
                    pose.extend(torch.split(fpose, window_length))

        self.imu = imu
        self.pose = pose

    def __getitem__(self, idx):
        _imu = self.imu[idx].float()
        _pose = self.pose[idx].float()
        _input = _imu

        if self.config.r6d == True:
            _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
        else:
            _output = _pose

        return _input, _output

    def __len__(self):
        return len(self.imu)

