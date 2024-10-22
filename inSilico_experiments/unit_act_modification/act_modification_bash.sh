#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --array=1-8
#SBATCH --mail-user=alireza@hms.harvard.edu
#SBATCH -o unit_modif_small_set_evol_%A_%a.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 128 --score_method  MSE --reps 2 --reps_samlping 1 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 128 --score_method  cosine --reps 2 --reps_samlping 1 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 128 --score_method  MSE --reps 2 --reps_samlping 1 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 128 --score_method  cosine --reps 2 --reps_samlping 1 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 256 --score_method  MSE --reps 2 --reps_samlping 1 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 256 --score_method  cosine --reps 2 --reps_samlping 1 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 256 --score_method  MSE --reps 2 --reps_samlping 1 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 256 --score_method  cosine --reps 2 --reps_samlping 1 --steps 100 --sampling_strategy random
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"


module load gcc/9.2.0
module load cuda/11.7
module load miniconda3/4.10.3

source activate cosine-project-O2

cd ~/Cosine-Project/inSilico_experiments/unit_act_modification
python3 unit_act_modification_eval.py $unit_name
