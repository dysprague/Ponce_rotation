#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=73-108
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o cosine_evol_%j.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 500 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 1024 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 500 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 2048 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 2048 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 2048 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 1024 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 1024 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 1024 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 200  --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 200  --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 200  --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 100  --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 100  --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer4.Bottleneck2 --popsize 100  --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize  50 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize  50 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize  50 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_re
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 100 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 100 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 100 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 200 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 200 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer2.Bottleneck2 --popsize 200 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize  50 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize  50 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize  50 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 100 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 0 50 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 50 100 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
--net resnet50 --G fc6 --layer .layer3.Bottleneck0 --popsize 200 --pop_rand_seed 0 --target_idx 100 150 --score_method cosine corr MSE dot --reps 3 --RFresize --resize_ref
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/Cosine-Project
python3 inSilico_experiments/cosine_evol_O2_cluster.py $unit_name
