
## series of in silico experiment for population reconstruction 

## 1. Introduction

## 2. DEVELOPERS
    scripts were developed by following packages:
    - python 3.10
    - Cuda 11.7
    - cuDnn v8.8.1 
    - Pytorch 2
    (other packes is listed in requirements.txt)
    
be careful to install the correct version of packages. python + cuda + cudnn + pytorch
version should be compatible with each other. Look this link for more information:
https://pub.towardsai.net/installing-pytorch-with-cuda-support-on-windows-10-a38b1134535e
https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu

## 3. O2 cluster 
do this commands in O2 cluster:
```shell
srun -n 1 --pty -t 3:00:00 -p gpu_quad --gres=gpu:1 --mem=8G bash
module load gcc/9.2.0 cuda/11.7 miniconda3/4.10.3
source activate cosine-project-O2
```
