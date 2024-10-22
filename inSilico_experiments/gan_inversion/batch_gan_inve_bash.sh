#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --array=1-12
#SBATCH --mail-user=alireza@hms.harvard.edu
#SBATCH -o gan_inve_%A_%a.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--folder_name monkey --batch_num 1
--folder_name monkey --batch_num 2
--folder_name monkey --batch_num 3
--folder_name monkey --batch_num 4
--folder_name monkey --batch_num 5
--folder_name monkey --batch_num 6
--folder_name non_monkey --batch_num 1
--folder_name non_monkey --batch_num 2
--folder_name non_monkey --batch_num 3
--folder_name non_monkey --batch_num 4
--folder_name non_monkey --batch_num 5
--folder_name non_monkey --batch_num 6
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"


module load gcc/9.2.0
module load cuda/11.7
module load miniconda3/4.10.3

source activate cosine-project-O2

cd ~/Cosine-Project/inSilico_experiments/gan_invertion
python3 batch_gan_inversion.py $unit_name
