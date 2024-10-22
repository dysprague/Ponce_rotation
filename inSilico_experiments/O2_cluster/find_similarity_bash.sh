#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-user=alireza@hms.harvard.edu
#SBATCH -o find_similarity_%j.out

module load gcc/9.2.0
module load cuda/11.7
module load miniconda3/4.10.3

source activate cosine-project-O2

cd ~/Cosine-Project/inSilico_experiments/O2_cluster
python3 find_similarity_o2.py

