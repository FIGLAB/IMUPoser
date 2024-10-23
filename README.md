# IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds
Click to watch the video!
<p align="center">
  <a href="https://www.youtube.com/watch?v=hgpjbKv8XFY"><img src="media/IMUPoser_github.png" alt="animated" width="100%"/></a>
</p>

Research code for IMUPoser (CHI 2023)

## Reference
Vimal Mollyn, Riku Arakawa, Mayank Goel, Chris Harrison, and Karan Ahuja. 2023. IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (CHI '23). Association for Computing Machinery, New York, NY, USA, Article 529, 1–12.

[Download Paper Here](https://drive.google.com/uc?export=download&id=1FYB52VN_v3ZIh99BNVLffXzqHRyD23rG)


BibTeX Reference:

```
@inproceedings{10.1145/3544548.3581392,
author = {Mollyn, Vimal and Arakawa, Riku and Goel, Mayank and Harrison, Chris and Ahuja, Karan},
title = {IMUPoser: Full-Body Pose Estimation Using IMUs in Phones, Watches, and Earbuds},
year = {2023},
isbn = {9781450394215},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3544548.3581392},
doi = {10.1145/3544548.3581392},
booktitle = {Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems},
articleno = {529},
numpages = {12},
keywords = {sensors, inertial measurement units, mobile devices, Motion capture},
location = {Hamburg, Germany},
series = {CHI '23}
}
```

## 1. Clone (or Fork!) this repository
```
git clone https://github.com/FIGLAB/IMUPoser.git
git clone https://github.com/bryanbocao/IMUPoser.git
```
 
## 2. Create a virtual environment
We recommend using conda. Tested on `Ubuntu 18.04`, with `python 3.7`.

```bash
conda create -n "imuposer" python=3.7
conda activate imuposer
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

python -m pip install -r requirements.txt
python -m pip install -e src/
```

## 3. Download training data
1. Download training data from [AMASS](https://amass.is.tue.mpg.de/index.html) and [DIP-IMU](https://dip.is.tuebingen.mpg.de/). We use ```<ROOT>``` to refer to the root path of this repository in your file system.
Prepare folders in the following format:
```
<ROOT>
  └── data
    └── raw
        ├── AMASS
          ├── tar
        ├── DIP_IMU
          ├── zip
```
Below is an example of scripts to generate these folders in Command Line Interface (CLI):
```
IMUPoser$
mkdir data
cd data/
mkdir raw
cd raw
mkdir datasets
cd datasets
mkdir AMASS
cd AMASS
mkdir tar
cd tar
```

Register an account both websites below.

**AMASS** (SMPL-H) https://amass.is.tue.mpg.de/download.php. Download all datasets with the ```SMPL+H G``` (or ```SMPL-X G``` for CNRS) format into this folder
```<ROOT>/data/raw/datasets/AMASS/tar```. Note BMLrub, GRAB, SOMA, WEIZMANN are not used.

**DIP_IMU** (DIP IMU AND OTHERS - DOWNLOAD SERVER 1) https://dip.is.tuebingen.mpg.de/index.html.

The files should follow the structure below:
```
<ROOT>
  └── data
    └── raw
        ├── AMASS
          ├── tar
            ├── ACCAD.tar.bz2              (ACCAD - SMPL+H G)
            ├── (None)                     (BMLhandball - SMPL+H G)
            ├── BMLmovi.tar.bz2            (BMLmovi - SMPL+H G)
            ├── CMU.tar.bz2                (CMU - SMPL+H G)
            ├── CNRS.tar.bz2               (CNRS - SMPL-X G)
            ├── DFaust.tar.bz2             (DFaust - SMPL+H G)
            ├── DanceDB.tar.bz2            (DanceDB - SMPL+H G)
            ├── EKUT.tar.bz2               (EKUT - SMPL+H G)
            ├── EyesJapanDataset.tar.bz2   (EyesJapanDataset - SMPL+H G)
            ├── HDM05.tar.bz2              (HDM05 - SMPL+H G)
            ├── HUMAN4D.tar.bz2            (HUMAN4D - SMPL+H G)
            ├── HumanEva.tar.bz2           (HumanEva - SMPL+H G)
            ├── KIT.tar.bz2                (KIT - SMPL+H G)
            ├── MoSh.tar.bz2               (MoSh - SMPL+H G)
            ├── PosePrior.tar.bz2          (PosePrior - SMPL+H G)
            ├── SFU.tar.bz2                (SFU - SMPL+H G)
            ├── SSM.tar.bz2                (SSM - SMPL+H G)
            ├── TCDHands.tar.bz2           (TCDHands - SMPL+H G)
            ├── TotalCapture.tar.bz2       (TotalCapture - SMPL+H G)
            ├── Transitions.tar.bz2        (Transitions - SMPL+H G)
        ├── DIP_IMU
          ├── zip
            ├── DIPIMUandOthers.zip
```

Follow this structure:
```
<ROOT>
  └── data
    └── raw
      ├── AMASS
      │   ├── ACCAD
      │   ├── BioMotionLab_NTroje(Unavailable from website)
      │   ├── BMLhandball
      │   ├── BMLmovi
      │   ├── CMU
      │   ├── DanceDB
      │   ├── DFaust_67
      │   ├── EKUT
      │   ├── Eyes_Japan_Dataset
      │   ├── HUMAN4D
      │   ├── HumanEva
      │   ├── KIT
      │   ├── MPI_HDM05
      │   ├── MPI_Limits
      │   ├── MPI_mosh
      │   ├── SFU
      │   ├── SSM_synced
      │   ├── TCD_handMocap
      │   ├── TotalCapture
      │   └── Transitions_mocap
      ├── DIP_IMU
      │   ├── s_01
      │   ├── s_02
      │   ├── s_03
      │   ├── s_04
      │   ├── s_05
      │   ├── s_06
      │   ├── s_07
      │   ├── s_08
      │   ├── s_09
      │   └── s_10
      └── README.md
```
Note ```BioMotionLab_NTroje``` is missing from the AMASS Dataset when I download the data: https://github.com/FIGLAB/IMUPoser/issues/12

Unzip all data:
```
cd <ROOT>/data/raw/datasets/AMASS/tar
for f in *.tar.bz2; do tar xf "$f"; done
(wait for a few minutes)
rm -r *.tar.bz2 (if you want to clean the zipped files)

cd <ROOT>/data/raw/datasets/DIP_IMU/zip
unzip DIPIMUandOthers.zip
cd DIP_IMU_and_Others
unzip DIP_IMU.zip

cd <ROOT>/data/raw
mv <ROOT>/data/raw/datasets/AMASS/tar/* <ROOT>/data/raw/AMASS
mv <ROOT>/data/raw/datasets/DIP_IMU/zip/DIP_IMU_and_Others/DIP_IMU <ROOT>/data/raw
```

Clean zipped data:
```
rm -r <ROOT>/data/raw/datasets/AMASS/tar
rm -r <ROOT>/data/raw/datasets/DIP_IMU/zip
```

## 4. Training Steps
1. Preprocess the AMASS and DIP-IMU datasets [scripts/1_preprocessing](scripts/1_preprocessing). Run all files in order.
2. Train the model [scripts/2_train/run_combos.sh](scripts/2_train/run_combos.sh) ([scripts/2_train/run_combos_long.sh](scripts/2_train/run_combos_long.sh) for long training).

## Disclaimer
```
THE PROGRAM IS DISTRIBUTED IN THE HOPE THAT IT WILL BE USEFUL, BUT WITHOUT ANY WARRANTY. IT IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW THE AUTHOR WILL BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
```

## Acknowledgments
Some of the modules in this repo were inspired by the amazing [TransPose](https://github.com/Xinyu-Yi/TransPose/) github repo. 

## License
The IMUPoser code can only be used for research i.e., non-commercial purposes. For a commercial license, please contact Vimal Mollyn, Karan Ahuja and Chris Harrison.

## Contact
Feel free to contact [Vimal Mollyn](mailto:ms123vimal@gmail.com) for any help, questions or general feedback!
