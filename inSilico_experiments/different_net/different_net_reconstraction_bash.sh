#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --array=1-186
#SBATCH --mail-user=alireza@hms.harvard.edu
#SBATCH -o different_nets_evol_%A_%a.out

echo "$SLURM_ARRAY_TASK_ID"

param_list=\
'--net resnet50_linf_8 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V1.BatchNorm2dnorm2 --layers_short v1 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d3 --layers_short conv2 --popsize 192 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V1.BatchNorm2dnorm2 --layers_short v1 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d3 --layers_short conv2 --popsize 192 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V1.BatchNorm2dnorm2 --layers_short v1 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d3 --layers_short conv2 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V1.BatchNorm2dnorm2 --layers_short v1 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d3 --layers_short conv2 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V1.BatchNorm2dnorm2 --layers_short v1 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d3 --layers_short conv2 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V1.BatchNorm2dnorm2 --layers_short v1 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d3 --layers_short conv2 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V1.BatchNorm2dnorm2 --layers_short v1 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d3 --layers_short conv2 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V1.BatchNorm2dnorm2 --layers_short v1 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d3 --layers_short conv2 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d3 --layers_short conv2 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer1.2.BatchNorm2dbn3 --layers_short layer1 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d3 --layers_short conv2 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 512 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 512 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V2.BatchNorm2dnorm3_1 --layers_short v2 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d6 --layers_short conv3 --popsize 384 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 512 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 512 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V2.BatchNorm2dnorm3_1 --layers_short v2 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d6 --layers_short conv3 --popsize 384 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V2.BatchNorm2dnorm3_1 --layers_short v2 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d6 --layers_short conv3 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V2.BatchNorm2dnorm3_1 --layers_short v2 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d6 --layers_short conv3 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V2.BatchNorm2dnorm3_1 --layers_short v2 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d6 --layers_short conv3 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V2.BatchNorm2dnorm3_1 --layers_short v2 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d6 --layers_short conv3 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V2.BatchNorm2dnorm3_1 --layers_short v2 --popsize 2 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d6 --layers_short conv3 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V2.BatchNorm2dnorm3_1 --layers_short v2 --popsize 2 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d6 --layers_short conv3 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 2 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 2 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d6 --layers_short conv3 --popsize 2 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 2 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer2.3.BatchNorm2dbn3 --layers_short layer2 --popsize 2 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d6 --layers_short conv3 --popsize 2 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 1024 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 1024 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V4.BatchNorm2dnorm3_3 --layers_short v4 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d8 --layers_short conv4 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 1024 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 1024 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V4.BatchNorm2dnorm3_3 --layers_short v4 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d8 --layers_short conv4 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V4.BatchNorm2dnorm3_3 --layers_short v4 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d8 --layers_short conv4 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V4.BatchNorm2dnorm3_3 --layers_short v4 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d8 --layers_short conv4 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V4.BatchNorm2dnorm3_3 --layers_short v4 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d8 --layers_short conv4 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V4.BatchNorm2dnorm3_3 --layers_short v4 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d8 --layers_short conv4 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V4.BatchNorm2dnorm3_3 --layers_short v4 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d8 --layers_short conv4 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V4.BatchNorm2dnorm3_3 --layers_short v4 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d8 --layers_short conv4 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .V4.BatchNorm2dnorm3_3 --layers_short v4 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d8 --layers_short conv4 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer3.5.BatchNorm2dbn3 --layers_short layer3 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .V4.BatchNorm2dnorm3_3 --layers_short v4 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d8 --layers_short conv4 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 2048 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 2048 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 512 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 2048 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 2048 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 512 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 256 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 512 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 512 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 512--score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 512 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 64 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 128 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 16 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 32 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 4 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 2 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 8 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 2 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net alexnet --layers .features.Conv2d10 --layers_short conv5 --popsize 1 --score_method MSE --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 2048 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 2048 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 512 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 2048 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 2048 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 512 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 512 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 512 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 128 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 512--score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 512 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 128 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 128 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 128 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 32 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 128 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 128 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 32 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 32 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 32 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 8 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 32 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 32 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 8 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 8 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 8 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 2 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy most
--net resnet50_linf_8 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 8 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net resnet50 --layers .layer4.2.BatchNorm2dbn3 --layers_short layer4 --popsize 8 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
--net cornet_s --layers .IT.BatchNorm2dnorm3_1 --layers_short IT --popsize 2 --score_method cosine --reps 5 --reps_samlping 5 --steps 100 --sampling_strategy random
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"


module load gcc/9.2.0
module load cuda/11.7
module load miniconda3/4.10.3

source activate cosine-project-O2

cd ~/Cosine-Project/inSilico_experiments/general_reconstructor
python3 fc6_general_reconstructor.py $unit_name
