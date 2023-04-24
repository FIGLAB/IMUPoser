# IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds
Research code for IMUPoser (CHI 2023)

## Reference
Vimal Mollyn, Riku Arakawa, Mayank Goel, Chris Harrison, and Karan Ahuja. 2023. IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (CHI '23). Association for Computing Machinery, New York, NY, USA, Article 529, 1–12. https://doi.org/10.1145/3544548.3581392

[Download Paper Here][https://drive.google.com/uc?export=download&id=1FYB52VN_v3ZIh99BNVLffXzqHRyD23rG]

BibTeX Reference:

```bash
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
```
 
## 2. Create a virtual environment
We recommend using conda. Tested on `Ubuntu 18.04`, with `python 3.7`.

```bash
conda create -n "imuposer" python=3.7
conda activate imuposer
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

python -m pip install -r requirements.txt
python -m pip install -e src/
```
