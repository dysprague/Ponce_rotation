@echo off
setlocal

rem Activate Conda environment
call conda activate cosine-project

rem Change to the directory where your Python script is located
cd C:\Users\Alireza\Documents\Git\Cosine-Project\inSilico_experiments\noisy_evol

rem Run the Python script with a single set of parameters

call python noisy_evol.py  --net alexnet --layers .features.ReLU1 --layers_short ReLU1 --reps 10 --steps 50 --noise no-noise
call python noisy_evol.py  --net alexnet --layers .features.ReLU1 --layers_short ReLU1 --reps 10 --steps 50 --noise noise
call python noisy_evol.py  --net alexnet --layers .features.ReLU4 --layers_short ReLU4 --reps 10 --steps 50 --noise no-noise
call python noisy_evol.py  --net alexnet --layers .features.ReLU4 --layers_short ReLU4 --reps 10 --steps 50 --noise noise
call python noisy_evol.py  --net alexnet --layers .features.ReLU7 --layers_short ReLU7 --reps 10 --steps 50 --noise no-noise
call python noisy_evol.py  --net alexnet --layers .features.ReLU7 --layers_short ReLU7 --reps 10 --steps 50 --noise noise
call python noisy_evol.py  --net alexnet --layers .features.ReLU9 --layers_short ReLU9 --reps 10 --steps 50 --noise no-noise
call python noisy_evol.py  --net alexnet --layers .features.ReLU9 --layers_short ReLU9 --reps 10 --steps 50 --noise noise
call python noisy_evol.py  --net alexnet --layers .features.ReLU11 --layers_short ReLU11 --reps 10 --steps 50 --noise no-noise
call python noisy_evol.py  --net alexnet --layers .features.ReLU11 --layers_short ReLU11 --reps 10 --steps 50 --noise noise
call python noisy_evol.py  --net alexnet --layers .classifier.ReLU2 --layers_short FC_ReLU2 --reps 10 --steps 50 --noise no-noise --sampling_mode sub --sub_sample_size 200
call python noisy_evol.py  --net alexnet --layers .classifier.ReLU2 --layers_short FC_ReLU2 --reps 10 --steps 50 --noise noise --sampling_mode sub --sub_sample_size 200
call python noisy_evol.py  --net alexnet --layers .classifier.ReLU5 --layers_short FC_ReLU5 --reps 10 --steps 50 --noise no-noise --sampling_mode sub --sub_sample_size 200
call python noisy_evol.py  --net alexnet --layers .classifier.ReLU5 --layers_short FC_ReLU5 --reps 10 --steps 50 --noise noise --sampling_mode sub --sub_sample_size 200
call python noisy_evol.py  --net alexnet --layers .classifier.Linear6 --layers_short FC_Linear6 --reps 10 --steps 50 --noise no-noise --sampling_mode sub --sub_sample_size 200
call python noisy_evol.py  --net alexnet --layers .classifier.Linear6 --layers_short FC_Linear6 --reps 10 --steps 50 --noise noise --sampling_mode sub --sub_sample_size 200

endlocal
