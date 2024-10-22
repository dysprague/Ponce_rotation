import sys
import os
import pandas as pd
from argparse import ArgumentParser
import warnings
#warnings.filterwarnings("ignore") # becairfull with this

parser = ArgumentParser()
parser.add_argument("--net", type=str, default="alexnet", help="Network model to use for Image distance computation")
parser.add_argument("--layers", type=str, default=".features.Conv2d10", nargs="+", 
                     help="Network model to use for Image distance computation")
parser.add_argument("--layers_short", type=str, default="conv5", help="shortcut for the layer name")
parser.add_argument("--popsize", type=int, default=256, help="Number of units in the population recording")
parser.add_argument("--sampling_strategy", type=str, default="random", choices=["random", "most"], help="select units randomly or based on their activation")
parser.add_argument("--input_size", type=tuple, default=(3, 227, 227), help="net input image size")
parser.add_argument("--img_size", type=tuple, default=(147, 147), help="image size after padding")
parser.add_argument("--pading_size", type=tuple, default=(40, 40), help="padding size")
parser.add_argument("--gan_name", type=str, default="fc6", help="Gan model to use for Image generation")
parser.add_argument("--score_method", type=str, default=["MSE"], nargs="+", 
    choices=["cosine", "Correlation", "MSE"], help="Objective function to assess the population response patteren similarity")
parser.add_argument("--steps", type=int, default=100, help="Num of evolution Steps")
parser.add_argument("--reps", type=int, default=5, help="Number of replications for each condition")
parser.add_argument("--reps_samlping", type=int, default=5, help="Number of replications of unit sampling")

if sys.platform == "linux":
    exp_result_main_root = r"/n/scratch3/users/a/ala2226/untrained_netsd_experiment_013023"
    refimgdir = r"/home/ala2226/data/big_data_set"
    sys.path.append(r"/home/ala2226/Cosine-Project") #TODO: the path to the codebase you are running on the cluster

else:
    exp_result_main_root = r"C:\Data\cosine\insilico_experiments\untrained_nets_experiment_090323_test3"
    refimgdir = r"C:\Data\cosine\insilico_experiments\data\big_data_set"
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
    popsize = args.popsize
    metric_name = args.score_method
    input_size = args.input_size
    img_size = args.img_size
    pading_size = args.pading_size
    NUM_OF_RUNS_PER_POP_SIZE = args.reps_samlping
    NUM_OF_RUNS_PER_IMAGE = args.reps

    if not isinstance(layer_name, list):
        raise ValueError("layer_name must be a list of strings")
    
    #%% make a dictionary of the parameters
    trial_param_dict = {"net_name": net_name,
                    "layer_name": layer_name,
                    "layer_short": layer_short,
                    "gan_name": gan_name,
                    "pop_size": popsize,
                    "input_size": input_size,
                    "pading_size": pading_size}
    #%% set up a pandas dataframe to save the inforamtions about each run
    expriment_meta_data_df = pd.DataFrame(columns=["output_type", "trget_imge_name", "similarity_metric", "pop_size", "pop_resampling_id",
                                       "gan_name", "layer_name", "layer_short", "net_name", "img_size", "pading_size", "input_size", "score", "init_score"
                                       "pop_unit_idx", "sub_pop_type", "gen_rerun_id"])
    
    #%% make name for expriment meta data frame file 
   
    # generate 5 digit random number for the meta data file name
    random_number = np.random.randint(10000, 99999)
    if len(metric_name) > 1:
        exp_file_name = f"data_{net_name}_{args.sampling_strategy}_{gan_name}_{layer_short}_{popsize}"
        meta_file_name = f"meta_data_{net_name}_{args.sampling_strategy}_{gan_name}_{layer_short}_{popsize}_{random_number}.h5"
    else:
        exp_file_name = f"data_{net_name}_{args.sampling_strategy}_{gan_name}_{layer_short}_{popsize}_{metric_name[0]}"
        meta_file_name = f"meta_data_{net_name}_{args.sampling_strategy}_{gan_name}_{layer_short}_{popsize}_{metric_name[0]}_{random_number}.h5"
    exp_result_root = os.path.join(exp_result_main_root, exp_file_name)
    # make a directory for the expriment
    os.makedirs(exp_result_root, exist_ok=True)
    # make the meta data directory
    os.makedirs(os.path.join(exp_result_main_root, "meta_data_files"), exist_ok=True)
    #%% Load the reference images
    refimgnms, refimgtsr = load_ref_imgs(
        imgdir=refimgdir, preprocess_type='center_crop', image_size=227)

    # %% set up the GAN
    G = upconvGAN(gan_name).cuda()
    G.requires_grad_(False)
    code_length = G.codelen

    #%% Run the experiment
    scorer = TorchScorer(net_name)
    module_names, module_types, module_spec = get_module_names(
        scorer.model, input_size, "cuda", False)
    #%% random population
    for imgid in range(len(refimgnms)):
        # Select target image and add target vector.
        targnm, target_imgtsr = refimgnms[imgid], refimgtsr[imgid:imgid + 1]
        # Set the population
        for pop_resampling_id in range(NUM_OF_RUNS_PER_POP_SIZE):
            # Set scorer and select population    
            if args.sampling_strategy == "random":
                unit_mask_dict, unit_tsridx_dict = set_random_population_recording(
                    scorer, layer_name, popsize=popsize, seed=pop_resampling_id)
                
            elif args.sampling_strategy == "most":
                unit_mask_dict, unit_tsridx_dict = set_most_active_population_recording(
                    scorer, layer_name, target_imgtsr, pading_size, img_size, popsize=popsize)
            else:
                raise NotImplementedError("Sampling strategy not implemented")    
            ## encoding the reference images, it's not efficient to do it her , but it works. TODO: fix this
            resf_images_actmat, ref_imgtsr_resized = encode_image(scorer, refimgtsr, key=layer_name,
                                    RFresize=True, corner=pading_size, imgsize=img_size)
            popul_m, popul_s = set_normalizer(resf_images_actmat)
            print(f"resf_images_actmat shape: {resf_images_actmat.shape}, popul_m shape: {popul_m.shape}, popul_s shape: {popul_s.shape}")
            print(f'popul_m: {popul_m}, popul_s: {popul_s}')
            # receptive field estimation
            fitdict = fr_estimatir(scorer, G, unit_tsridx_dict, layer_name, input_size,show_fig=False)
            targ_actmat, target_imgtsr_resized = encode_image(scorer, target_imgtsr, key=layer_name,
                                    RFresize=True, corner=pading_size, imgsize=img_size)
            targlabel = os.path.splitext(targnm)[0]
            # add the target image name to the dictionary
            trial_param_dict["trget_imge_name"] = targlabel
            trial_param_dict["pop_resampling_id"] = pop_resampling_id
            trial_param_dict["pop_unit_idx"] = unit_mask_dict
            trial_param_dict["sub_pop_type"] = args.sampling_strategy            

            target_imgtsr_resized_RF_masked =\
                (torch.from_numpy(np.absolute(fitdict["fitmap"][None,:,:])) / fitdict["fitmap"].max()) *\
                target_imgtsr_resized
            
            
            for score_method in metric_name:
                ## add the stuff to the dictionary
                trial_param_dict["similarity_metric"] = score_method
                # save the target\masked image
                print_tensor(target_imgtsr_resized[0], exp_result_root, expriment_meta_data_df, output_type="target_img", **trial_param_dict)
                print_tensor(target_imgtsr_resized_RF_masked[0], exp_result_root, expriment_meta_data_df, output_type="target_img_RF_masked", **trial_param_dict)
                for gen_rerun_id in range(NUM_OF_RUNS_PER_IMAGE):
                    ## add the generation re-run id to the dictionary
                    trial_param_dict["gen_rerun_id"] = gen_rerun_id
                    ## set fig title 
                    title_str = "%s-%s-%s-popsize%d-%s"%(trial_param_dict["trget_imge_name"],
                                            trial_param_dict["similarity_metric"], trial_param_dict["gan_name"],
                                            trial_param_dict["pop_size"], trial_param_dict["sub_pop_type"])
                    #set objective function
                    objfunc = set_objective(score_method, targ_actmat, popul_mask=None, normalize=True, popul_m=popul_m, popul_s=popul_s)
                    ## set optimizer
                    optimizer = CholeskyCMAES(code_length, population_size=None, init_sigma=3,
                                    init_code=np.zeros([1, code_length]), Aupdate_freq=10,
                                    maximize=True, random_seed=None, optim_params={})
                    
                    codes_all, scores_all, actmat_all, generations, last_gem_img, last_gem_img_maske, best_imgs, best_imgs_RF_masked, best_scores, best_codes =\
                        run_evol(scorer, objfunc, optimizer, G, reckey=layer_name, savedir=exp_result_root,
                                meta_data_df=expriment_meta_data_df, trial_param_dict=trial_param_dict,
                                titlestr =title_str, steps=args.steps, RFresize=True, corner=pading_size, imgsize=img_size,
                                RF_mask=fitdict["fitmap"], save_plot="full", save_data=True)
                    # 
                    figh = visualize_popul_act_evol(actmat_all, generations, targ_actmat, titlestr=title_str)
                    print_fig(figh, exp_result_root, expriment_meta_data_df, output_type= "popul_act_evol",**trial_param_dict) 

            # clear the memory
            scorer.cleanup()
            if args.sampling_strategy == "most":
                break
        #Let save the meta data frame to the disk in HDF5 format                       
        expriment_meta_data_df.to_hdf(os.path.join(exp_result_main_root, "meta_data_files", meta_file_name), key="expriment_meta_data_df", mode="w")
    #%% clear the memory
    torch.cuda.empty_cache() 