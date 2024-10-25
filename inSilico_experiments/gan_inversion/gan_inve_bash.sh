#!/bin/bash
#SBATCH --partition=kempner
#SBATCH --time=0-05:00
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-user=daniel_sprague@fas.harvard.edu
#SBATCH --output=gan_inve_7.out
#SBATCH --error=gan_inve_7.err

#param_list=\
#'--folder_name ambulance
#--folder_name cats_jumping
#--folder_name fan
#--folder_name horses 
#--folder_name komodo 
#--folder_name macaque_eating 
#--folder_name macaque_running 
#--folder_name macaque_fighting
#--folder_name monkey_grooming 
#--folder_name soccer_ball
#'

source /n/sw/Miniforge3-24.7.1-0/etc/profile.d/conda.sh

conda activate ponce_rotation

echo "installed packages:"
pip list

cd /n/home09/dsprague/Ponce_rotation/inSilico_experiments/gan_inversion
python3 batch_gan_inversion.py --task_id 7
