#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 0:15:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --array=1-204
#SBATCH --mail-user=alireza@hms.harvard.edu
#SBATCH -o prepreocessing_bigGAN_%A_%a.out

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--net alexnet --layers_short conv5 --popsize 1 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 2 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 4 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 8 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 16 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 32 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 64 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 128 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 256 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 1 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 2 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 4 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 8 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 16 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 32 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 64 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 128 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 256 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 1 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 2 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 4 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 8 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 16 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 32 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 64 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 128 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 256 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 1 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 2 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 4 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 8 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 16 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 32 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 64 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 128 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 256 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 1 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 2 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 4 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 8 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 16 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 32 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 64 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 128 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 256 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 384 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 1 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 2 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 4 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 8 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 16 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 32 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 64 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 128 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 256 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 384 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 1 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 2 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 4 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 8 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 16 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 32 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 64 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 128 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 192 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 1 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 2 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 4 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 8 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 16 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 32 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 64 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 128 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 192 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 1 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 2 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 4 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 8 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 16 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 32 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 64 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 128 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 256 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5 --popsize 1 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 2 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 4 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 8 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 16 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 32 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 64 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 128 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5 --popsize 256 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 1 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 2 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 4 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 8 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 16 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 32 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 64 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 128 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 256 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv4 --popsize 1 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 2 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 4 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 8 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 16 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 32 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 64 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 128 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv4 --popsize 256 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 1 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 2 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 4 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 8 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 16 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 32 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 64 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 128 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 256 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 384 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv3 --popsize 1 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 2 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 4 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 8 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 16 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 32 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 64 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 128 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 256 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv3 --popsize 384 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 1 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 2 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 4 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 8 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 16 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 32 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 64 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 128 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 192 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv2 --popsize 1 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 2 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 4 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 8 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 16 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 32 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 64 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 128 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv2 --popsize 192 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 1 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 1 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 1 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 1 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 2 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 2 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 2 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 2 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 4 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 4 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 4 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 4 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 8 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 8 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 8 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 8 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 16 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 16 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 16 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 16 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 32 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 32 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 32 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 32 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 64 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 64 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv5432 --popsize 64 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv5432 --popsize 64 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 2 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 2 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 2 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 2 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 4 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 4 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 4 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 4 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 8 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 8 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 8 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 8 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 16 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 16 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 16 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 16 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 32 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 32 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 32 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 32 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 64 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 64 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 64 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 64 --score_method cosine --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 128 --score_method MSE --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 128 --score_method cosine --sampling_strategy random
--net alexnet --layers_short conv53 --popsize 128 --score_method MSE --sampling_strategy most
--net alexnet --layers_short conv53 --popsize 128 --score_method cosine --sampling_strategy most
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"


module load gcc/9.2.0
module load cuda/11.7
module load miniconda3/4.10.3

source activate cosine-project-O2

cd ~/Cosine-Project/inSilico_experiments/BigGAN_reconstruction_O2
python3 O2_preprocessing.py $unit_name
