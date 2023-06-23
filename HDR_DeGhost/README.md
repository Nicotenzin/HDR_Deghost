# Code

```python
conda env list
conda activate "XXXXX"
conda deactivate
python -V
which python

输入输出尺寸大小不变
kernel_size = 1,stride = 1,padding = 0
kernel_size = 3,stride = 1,padding = 1
kernel_size = 5,stride = 1,padding = 2
kernel_size = 7,stride = 1,padding = 3


salloc（分配一个节点）
ssh + 节点名
exit  （退出节点）
exit （释放资源）
squeue
sbatch + xx.sh文件 （提交批处理作业）
scancel + xx.sh (取消作业)
identify + xxx.tif （查看图像尺寸大小）


\mathcal{L}
\mathcal{T}

pytorch常用代码
ghost_mask = torch.mean(ghost_mask, dim=1, keepdim=True)
求平均值


动态卷积层
from fightingcv_attention.conv.CondConv import *
m=CondConv(in_planes=, out_planes=, kernel_size=3, stride=1, padding=1)


tar -czf 文件名.tar.gz 待压缩的文件/目录
tar -xzf 文件名.tar.gz -C 目标路径
tar -tzf archive.tar.gz 查看压缩包都有哪些文件不会解压

conda list --name yyc
conda env remove -n 环境名称
conda list --name yyc
conda install -c conda-forge conda-pack
conda activate yyc
conda pack yyc
cp 源  目标

tmux

tmux list

tmux detach

tmux attatch -t 0

ctrl + d 删除tmux session

```

# Usage

## Requirements

* Python 3.7.0
* CUDA 10.0 on Ubuntu 18.04

Install the require dependencies:

```bash
conda create -n HDR_Deghost python=3.7
conda activate HDR_Deghost
pip install -r requirements.txt
```

## Dataset

1. Download the dataset (include the training set and test set) from [Kalantari17's dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
2. Move the dataset to `./data` and reorganize the directories as follows:

```
./data/Training
|--001
|  |--262A0898.tif
|  |--262A0899.tif
|  |--262A0900.tif
|  |--exposure.txt
|  |--HDRImg.hdr
|--002
...
./data/Test (include 15 scenes from `EXTRA` and `PAPER`)
|--001
|  |--262A2615.tif
|  |--262A2616.tif
|  |--262A2617.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
|--BarbequeDay
|  |--262A2943.tif
|  |--262A2944.tif
|  |--262A2945.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
```

3. Prepare the corpped training set by running:

```
cd ./dataset
python gen_crop_data.py --patch_size=256 --stride=64 --aug
# aug: flip and rotate to expand the dataset
```

### Training & Evaluaton

```
cd HDR_Deghost
```

To train the model, run:

```
python train.py
```

To evaluate, run:

```
python test.py
```

