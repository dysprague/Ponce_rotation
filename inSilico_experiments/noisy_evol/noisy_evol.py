# first we evolve to reconstruct an image and then we will try to suppress all single units

import sys
import os
import pandas as pd
from argparse import ArgumentParser
import numpy as np
from torchvision.transforms import ToPILImage
#import warnings
#warnings.filterwarnings("ignore") # becairfull with this

parser = ArgumentParser()
parser.add_argument("--net", type=str, default="alexnet", help="Network model to use for Image distance computation")
parser.add_argument("--layers", type=str, default=".features.Conv2d10", help="Network model to use for Image distance computation")
parser.add_argument("--layers_short", type=str, default="conv5", help="shortcut for the layer name")
parser.add_argument("--sampling_mode", type=str, default="all", choices=["all", "sub"], help="select all units or a subsample of them")
parser.add_argument("--gan_name", type=str, default="fc6", help="Gan model to use for Image generation")
parser.add_argument("--steps", type=int, default=50, help="Num of evolution Steps")
parser.add_argument("--reps", type=int, default=10, help="Number of replications per unti")
parser.add_argument("--noise", type=str, default="noise", choices=["noise", "no-noise"], help="noise or no noise in evolution")
parser.add_argument("--sub_sample_size", type=int, default=200, help="Number of units to sample from the layer")

if sys.platform == "linux":
    exp_result_main_root = r"/n/scratch/users/a/ala2226/noisy_evol"
    sys.path.append(r"/home/ala2226/Cosine-Project") 
else:
    exp_result_main_root = r"N:\PonceLab\Users\Alireza\insilico_experiments\noisy_evol"
    sys.path.append(r"C:\Users\Alireza\Documents\Git\Cosine-Project")

os.makedirs(exp_result_main_root, exist_ok=True)

if __name__=="__main__":
    #%%
    #import datetime
    from core.utils.GAN_utils import upconvGAN
    from core.utils.Optimizers import CholeskyCMAES
    from core.utils.CNN_scorers import TorchScorer
    from core.utils.func_lib import *
    #%% 
    # Set parameters
    args = parser.parse_args() 

    net_name = args.net
    layer_name = args.layers
    layer_short = args.layers_short
    gan_name = args.gan_name
    num_runs = args.reps
    sampling_mode = args.sampling_mode
    steps = args.steps
    noise_flag = args.noise
    sub_sample_size = args.sub_sample_size
    # let print layer name and layer short name
    print(f"layer name: {layer_name}, layer short name: {layer_short}, noise flag: {noise_flag}")

    # generate 5 digit random number for the meta data file name
    random_number = np.random.randint(10000, 99999)
    exp_file_name = f"data_{net_name}_{sampling_mode}_{gan_name}_{layer_short}_{noise_flag}"

    exp_result_root = os.path.join(exp_result_main_root, exp_file_name)
    # make a directory for the expriment
    os.makedirs(exp_result_root, exist_ok=True)
    # %% set up the GAN
    G = upconvGAN(gan_name).cuda().eval()
    G.requires_grad_(False)
    code_length = G.codelen

    #%% Run the experiment
    scorer = TorchScorer(net_name)
    #%% let run the experiment
    # selet the unit
    if layer_name == ".features.ReLU1":
        # let have all units in the cenral layer (0:63, 23, 23)
        # add position of the unit in the layer
        all_neurons = [(i, 23, 23) for i in range(64)]
    elif layer_name == ".features.ReLU4":
        all_neurons = [(i, 14, 14) for i in range(192)]
    elif layer_name == ".features.ReLU7":
        all_neurons = [(i, 7, 7) for i in range(384)]
    elif layer_name == ".features.ReLU9":
        all_neurons = [(i, 7, 7) for i in range(256)]
    elif layer_name == ".features.ReLU11":
        all_neurons = [(i, 7, 7) for i in range(256)]
    elif layer_name == ".classifier.ReLU2":
        all_neurons = [(i,) for i in range(4096)]
    elif layer_name == ".classifier.ReLU5":
        all_neurons = [(i,) for i in range(4096)]
    elif layer_name == ".classifier.Linear6":
        all_neurons = [(i,) for i in range(1000)]
    else:
        ValueError ("layer name is not correct")

    print(f'sub samplin mode is {sampling_mode}')
    if sampling_mode == 'sub':
        # let select a random sample of the units
        np.random.seed(1234)
        print(f"nun of units in the layer is {np.shape(all_neurons)[0]}" )
        if np.shape(all_neurons)[0] > sub_sample_size:
            random_indices = np.random.choice(np.shape(all_neurons)[0], size=int(sub_sample_size), replace=False)
            all_neurons = [all_neurons[i] for i in random_indices]
            print(f"nun of units in the layer is {np.shape(all_neurons)[0]}")
    for neuron in all_neurons:
            # let set the unit
        h = scorer.set_unit("score", layer_name, neuron, ingraph=False) 

        # Number of runs you want to perform and steps per run
        noise_std_dev_base = np.exp(1)  # Constant scaling factor

        # Initialize lists to hold the aggregated data for each run
        all_means_per_run = []
        all_mean_std_devs_per_run = []
        all_mean_noisy_scores_per_run = []

        # Initialize a DataFrame to hold all individual scores and standard deviations for each run
        all_data = []

        # file name for the detailed data
        detailed_file_name_root = f'{layer_short}_{neuron[0]:03d}_detail_data'
        mean_file_name = f'{layer_short}_{neuron[0]:03d}_mean_data'
        for run in range(num_runs):
            # Initialize lists to hold the step data for the current run
            scores_all_pre_noise = []
            std_devs_all = []
            noisy_scores_all = []
            latent_code_best_noisy_all = []

            pre_noise_score_all_img = []
            noise_score_all_img = []

            init_laten_code = np.random.randn(1, code_length)
            new_codes_noisy = init_laten_code
            optimizer = CholeskyCMAES(space_dimen=code_length, init_code=new_codes_noisy, init_sigma=3.0,)


            for step in range(steps):
                latent_code_noisy = torch.from_numpy(np.array(new_codes_noisy)).float()
                imgs_noisy = G.visualize(latent_code_noisy.cuda()).cpu()
                scores_pre_noise = scorer.score_tsr(imgs_noisy)
                
                if noise_flag == "no-noise":
                    scores_noisy = scores_pre_noise
                    scaled_std_devs = np.zeros_like(scores_pre_noise)
                elif noise_flag == "noise":
                    scaled_std_devs = noise_std_dev_base * np.power(scores_pre_noise, 0.5)
                    scores_noisy = np.array([np.random.normal(score, scale) for score, scale in zip(scores_pre_noise, scaled_std_devs)])
                else:
                    raise ValueError("Noise flag must be either 'noise' or 'no noise'")
                
                best_immage_id = np.argmax(scores_noisy)
                best_image_this_step = new_codes_noisy[best_immage_id]


                new_codes_noisy = optimizer.step_simple(scores_noisy, new_codes_noisy)
                # get code coresponded to the best image
                
                # Append the aggregated data to the lists
                mean_score = np.mean(scores_pre_noise)
                scores_all_pre_noise.append(mean_score)
                mean_std_dev = np.mean(scaled_std_devs)
                std_devs_all.append(mean_std_dev)
                mean_noisy_score = np.mean(scores_noisy)
                noisy_scores_all.append(mean_noisy_score)
                latent_code_best_noisy_all.append(best_image_this_step)
                pre_noise_score_all_img.append(scores_pre_noise)
                noise_score_all_img.append(scores_noisy)

            # let have images for the last step
            mean_image = imgs_noisy.mean(dim=0).squeeze().detach().cpu()
            best_immage_id = np.argmax(scores_noisy)
            best_image = imgs_noisy[best_immage_id].squeeze().detach().cpu()


            # save the images
            mean_image_name = f'{layer_short}_{neuron[0]:03d}_mean_image_run_{run:02d}.png'
            mean_image_path = os.path.join(exp_result_root, mean_image_name)
            best_image_name = f'{layer_short}_{neuron[0]:03d}_best_image_run_{run:02d}.png'
            best_image_path = os.path.join(exp_result_root, best_image_name)
            ToPILImage()(mean_image).save(mean_image_path)
            ToPILImage()(best_image).save(best_image_path)




            # Append the aggregated run data to the respective lists
            all_means_per_run.append(scores_all_pre_noise)
            all_mean_std_devs_per_run.append(std_devs_all)
            all_mean_noisy_scores_per_run.append(noisy_scores_all)

            # let save the data as NPZ file for each run
            detailed_file_name = f'{detailed_file_name_root}_run_{run:02d}.npz'
            detailed_file_save_path = os.path.join(exp_result_root, detailed_file_name)
            np.savez(detailed_file_save_path, layer_name=layer_name ,layer_short=layer_short, neuron=neuron, run=run, steps=steps, init_laten_code=init_laten_code, noise_flag = noise_flag,
                        latent_code_best_noisy_all=latent_code_best_noisy_all,
                        pre_noise_score_all_img=pre_noise_score_all_img,
                        noise_score_all_img=noise_score_all_img)


        # Create DataFrames for both the detailed and aggregated data
        noisy_means_df = pd.DataFrame({
            'Run': np.repeat(np.arange(num_runs), steps),
            'Step': np.tile(np.arange(steps), num_runs),
            'Mean_Pre_Noise_Score': np.concatenate(all_means_per_run),
            'Mean_Std_Dev': np.concatenate(all_mean_std_devs_per_run),
            'Mean_Noisy_Score': np.concatenate(all_mean_noisy_scores_per_run),
            'unitID':  np.repeat(neuron[0], num_runs*steps),
            'layer_name': np.repeat(layer_name, num_runs*steps),
            'layer_short': np.repeat(layer_short, num_runs*steps),
            'noise_flag': np.repeat(noise_flag, num_runs*steps)
        })

        # Save both DataFrames to CSV files
        noisy_means_name = f'{mean_file_name}.csv'
        noisy_means_path = os.path.join(exp_result_root, noisy_means_name)
        noisy_means_df.to_csv(noisy_means_path, index=False)
        # let save as hdf5 file
        noisy_means_name_hdf5 = f'{mean_file_name}.h5'
        noisy_means_path_hdf5 = os.path.join(exp_result_root, noisy_means_name_hdf5)
        noisy_means_df.to_hdf(noisy_means_path_hdf5, key="noisy_means_df", mode="w")

        scorer.cleanup()
        #let save the image
