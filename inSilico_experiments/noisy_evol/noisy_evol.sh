#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --array= 1-16
#SBATCH --mail-user=alireza@hms.harvard.edu
#SBATCH -o noisy_evol_%A_%a.out

param_list=\
'--net alexnet --layers .features.ReLU1 --layers_short ReLU1 --reps 10 --steps 50 --noise no-noise
--net alexnet --layers .features.ReLU1 --layers_short ReLU1 --reps 10 --steps 50 --noise noise
--net alexnet --layers .features.ReLU4 --layers_short ReLU4 --reps 10 --steps 50 --noise no-noise
--net alexnet --layers .features.ReLU4 --layers_short ReLU4 --reps 10 --steps 50 --noise noise
--net alexnet --layers .features.ReLU7 --layers_short ReLU7 --reps 10 --steps 50 --noise no-noise
--net alexnet --layers .features.ReLU7 --layers_short ReLU7 --reps 10 --steps 50 --noise noise
--net alexnet --layers .features.ReLU9 --layers_short ReLU9 --reps 10 --steps 50 --noise no-noise
--net alexnet --layers .features.ReLU9 --layers_short ReLU9 --reps 10 --steps 50 --noise noise
--net alexnet --layers .features.ReLU11 --layers_short ReLU11 --reps 10 --steps 50 --noise no-noise
--net alexnet --layers .features.ReLU11 --layers_short ReLU11 --reps 10 --steps 50 --noise noise
--net alexnet --layers .classifier.ReLU2 --layers_short FC_ReLU2 --reps 10 --steps 50 --noise no-noise --sampling_mode sub --sub_sample_size 200
--net alexnet --layers .classifier.ReLU2 --layers_short FC_ReLU2 --reps 10 --steps 50 --noise noise --sampling_mode sub --sub_sample_size 200
--net alexnet --layers .classifier.ReLU5 --layers_short FC_ReLU5 --reps 10 --steps 50 --noise no-noise --sampling_mode sub --sub_sample_size 200
--net alexnet --layers .classifier.ReLU5 --layers_short FC_ReLU5 --reps 10 --steps 50 --noise noise --sampling_mode sub --sub_sample_size 200
--net alexnet --layers .classifier.Linear6 --layers_short FC_Linear6 --reps 10 --steps 50 --noise no-noise --sampling_mode sub --sub_sample_size 200
--net alexnet --layers .classifier.Linear6 --layers_short FC_Linear6 --reps 10 --steps 50 --noise noise --sampling_mode sub --sub_sample_size 200
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"


module load gcc/9.2.0
module load cuda/11.7
module load miniconda3/4.10.3

source activate cosine-project-O2

cd ~/Cosine-Project/inSilico_experiments/noisy_evol
python3 noisy_evol.py $unit_name
