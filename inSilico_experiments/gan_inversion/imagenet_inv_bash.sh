#!/bin/bash
#SBATCH --partition=kempner
#SBATCH --time=0-08:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=daniel_sprague@fas.harvard.edu
#SBATCH --output=gan_inve_%a.out
#SBATCH --error=gan_inve_%a.err

source /n/sw/Miniforge3-24.7.1-0/etc/profile.d/conda.sh

conda activate ponce_rotation

cd /n/home09/dsprague/Ponce_rotation/inSilico_experiments/gan_inversion
python3 imagenet_encode.py 
