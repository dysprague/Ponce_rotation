#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 00:00:10
#SBATCH --gres=gpu:1
#SBATCH --mail-user=alireza@hms.harvard.edu
#SBATCH -o env_test_%j.out


module load gcc/9.2.0
module load cuda/11.7

source activate cosine-project-O2

cd ~/Cosine-Project/inSilico_experiments/BigGAN_reconstruction_O2
python3 test_cluster_evn.py 
