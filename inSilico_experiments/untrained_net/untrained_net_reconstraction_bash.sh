#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --array=1-8
#SBATCH --mail-user=alireza@hms.harvard.edu
#SBATCH -o untrained_neet_evol_%A_%a.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--net alexnet_untrained --layers .features.Conv2d10 .features.Conv2d8 .features.Conv2d6 .features.Conv2d3 --layers_short conv5432 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet_untrained --layers .features.Conv2d10 .features.Conv2d8 .features.Conv2d6 .features.Conv2d3 --layers_short conv5432 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet_untrained --layers .features.Conv2d10 .features.Conv2d8 .features.Conv2d6 .features.Conv2d3 --layers_short conv5432 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet_untrained --layers .features.Conv2d10 .features.Conv2d8 .features.Conv2d6 .features.Conv2d3 --layers_short conv5432 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet_untrained --layers .features.Conv2d10 .features.Conv2d8 .features.Conv2d6 .features.Conv2d3 --layers_short conv5432 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet_untrained --layers .features.Conv2d10 .features.Conv2d8 .features.Conv2d6 .features.Conv2d3 --layers_short conv5432 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet_untrained --layers .features.Conv2d10 .features.Conv2d8 .features.Conv2d6 .features.Conv2d3 --layers_short conv5432 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet_untrained --layers .features.Conv2d10 .features.Conv2d8 .features.Conv2d6 .features.Conv2d3 --layers_short conv5432 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
'
export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"


module load gcc/9.2.0
module load cuda/11.7
module load miniconda3/4.10.3

source activate cosine-project-O2

cd ~/Cosine-Project/inSilico_experiments/general_reconstructor
python3 fc6_general_reconstructor.py $unit_name
