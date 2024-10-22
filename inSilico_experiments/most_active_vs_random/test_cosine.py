#%% 
# Importing libraries 
from datetime import datetime
import os
import pandas as pd
import sys
sys.path.append(r"C:\Users\Alireza\Documents\Git\Cosine-Project")
from core.utils.func_lib import *
import matplotlib.pylab as plt
##
import warnings
warnings.filterwarnings("ignore") # becairfull with this
#%%
# parameters parsing
# 
NUM_OF_RUNS_PER_POP_SIZE =6
NUM_OF_RUNS_PER_IMAGE = 4
#%%
# set main fynction
if __name__=="__main__":
    #%%
    #import datetime
    now = datetime.now()
    from core.utils.GAN_utils import upconvGAN
    from core.utils.Optimizers import CholeskyCMAES
    from core.utils.CNN_scorers import TorchScorer, resize_and_pad_tsr
    #%% 
    # Set parameters
    date_time_str = now.strftime("%Y-%m-%d_%H-%M")
    refimgdir = r"C:\Data\cosine\insilico_experiments\papulation_size_effect\data\data_sample"
    exp_result_root = r"C:\Data\cosine\insilico_experiments\Most_active_vs_random\results\real_ten_images%s" % date_time_str
    os.makedirs(exp_result_root, exist_ok=True)

    net_name = "alexnet"
    layer_name = ".features.Conv2d10"
    layer_short = "conv5"
    gan_name = "fc6"
    pop_size = [256]
    metric_name = ["cosine"]
    input_size = (3, 227, 227)
    img_size = (187, 187)
    pading_size = (20, 20) 
    #%% make a dictionary of the parameters
    param_dict = {"net_name": net_name,
                    "layer_name": layer_name,
                    "layer_short": layer_short,
                    "gan_name": gan_name,
                    "pop_size": pop_size,
                    "input_size": input_size,
                    "pading_size": pading_size}
    trial_param_dict = param_dict.copy()
    #%% set up a pandas dataframe to save the inforamtions about each run
    expriment_meta_data_df = pd.DataFrame(columns=["output_type", "trget_imge_name", "similarity_metric", "pop_size", "pop_resampling_id",
                                       "gan_name", "layer_name", "layer_short", "net_name", "img_size", "pading_size", "input_size", "score",
                                       "pop_unit_idx", "sub_pop_type",
                                       "gen_rerun_id"])

    #%% save the parameters
    #%% Load the reference images
    refimgnms, refimgtsr = load_ref_imgs(
        imgdir=refimgdir, preprocess_type='center_crop', image_size=227)

    # %% set up the GAN
    G = upconvGAN(gan_name).cuda()
    G.requires_grad_(False)
    code_length = G.codelen

    #%% Run the experiment
    scorer_random = TorchScorer(net_name)
    scorer_most = TorchScorer(net_name)
    scorer_least = TorchScorer(net_name)
    module_names, module_types, module_spec = get_module_names(
        scorer_random.model, input_size, "cuda", False)
    for popsize in pop_size:
        # add the population size to the dictionary
        pop_param_dict = param_dict.copy()
        pop_param_dict["pop_size"] = popsize
        #%% random population
        imgid = 9
        # Select target image and add target vector.
        targnm, target_imgtsr = refimgnms[imgid], refimgtsr[imgid:imgid + 1]
        # Set the population
        for pop_resampling_id in range(NUM_OF_RUNS_PER_POP_SIZE):
            # Set scorer and select population    
            unit_mask_dict_random, unit_tsridx_dict_random = set_random_population_recording(
                scorer_random, [layer_name], popsize=popsize, seed=pop_resampling_id)
            # receptive field estimation
            fitdict_random = fr_estimatir(scorer_random, G, unit_tsridx_dict_random, layer_name, input_size,show_fig=False)
            targ_actmat, target_imgtsr_resized = encode_image(scorer_random, target_imgtsr, key=layer_name,
                                    RFresize=True, corner=pading_size, imgsize=img_size)
            targlabel = os.path.splitext(targnm)[0]
            # add the target image name to the dictionary
            trial_param_dict = pop_param_dict.copy()
            trial_param_dict["trget_imge_name"] = targlabel
            trial_param_dict["pop_resampling_id"] = pop_resampling_id
            trial_param_dict["pop_unit_idx"] = unit_mask_dict_random[layer_name]
            trial_param_dict["sub_pop_type"] = "random"
            # save the target\masked image
            print_tensor(target_imgtsr_resized[0], exp_result_root, expriment_meta_data_df, output_type="target_img", **trial_param_dict)

            target_imgtsr_resized_RF_masked =\
                (torch.from_numpy(np.absolute(fitdict_random["fitmap"][None,:,:])) / fitdict_random["fitmap"].max()) *\
                target_imgtsr_resized
            print_tensor(target_imgtsr_resized_RF_masked[0], exp_result_root, expriment_meta_data_df, output_type="target_img_RF_masked", **trial_param_dict)
            
            for score_method in metric_name:
                ## add the stuff to the dictionary
                trial_param_dict["similarity_metric"] = score_method
                for gen_rerun_id in range(NUM_OF_RUNS_PER_IMAGE):
                    ## add the generation re-run id to the dictionary
                    trial_param_dict["gen_rerun_id"] = gen_rerun_id
                    ## set fig title 
                    title_str = "%s-%s-%s-popsize%d-random"%(trial_param_dict["trget_imge_name"],
                                            trial_param_dict["similarity_metric"], trial_param_dict["gan_name"],
                                            trial_param_dict["pop_size"])
                    #set objective function
                    objfunc = set_objective(score_method, targ_actmat, popul_mask=None, normalize=False)
                    ## set optimizer
                    optimizer = CholeskyCMAES(code_length, population_size=None, init_sigma=3,
                                    init_code=np.zeros([1, code_length]), Aupdate_freq=10,
                                    maximize=True, random_seed=None, optim_params={})
                    
                    codes_all, scores_all, actmat_all, generations, last_gem_img, last_gem_img_maske, best_imgs, best_imgs_RF_masked, best_scores =\
                        run_evol(scorer_random, objfunc, optimizer, G, reckey=layer_name, savedir=exp_result_root, meta_data_df=expriment_meta_data_df, trial_param_dict=trial_param_dict,
                        titlestr =title_str, steps=100, RFresize=True, corner=pading_size, imgsize=img_size, RF_mask=fitdict_random["fitmap"], save_plot="full", save_data=True)
                    # 
                    figh = visualize_popul_act_evol(actmat_all, generations, targ_actmat, titlestr=title_str)
                    print_fig(figh, exp_result_root, expriment_meta_data_df, output_type= "popul_act_evol",**trial_param_dict)                                               
            # clear the memory
            scorer_random.cleanup()
        #%% most active population
        imgid = 9
        # Select target image and add target vector.
        targnm, target_imgtsr = refimgnms[imgid], refimgtsr[imgid:imgid + 1]
        # Set the population
        # Set scorer and select population    
        unit_mask_dict_most, unit_tsridx_dict_most = set_most_active_population_recording(
            scorer_most, [layer_name], target_imgtsr, pading_size, img_size, popsize=popsize)
        # receptive field estimation
        fitdict_most= fr_estimatir(scorer_most, G, unit_tsridx_dict_most, layer_name, input_size,show_fig=False)
        targ_actmat, target_imgtsr_resized = encode_image(scorer_most, target_imgtsr, key=layer_name,
                                RFresize=True, corner=pading_size, imgsize=img_size)
        targlabel = os.path.splitext(targnm)[0]
        # add the target image name to the dictionary
        trial_param_dict = pop_param_dict.copy()
        trial_param_dict["trget_imge_name"] = targlabel
        trial_param_dict["pop_unit_idx"] = unit_mask_dict_most[layer_name]
        trial_param_dict["sub_pop_type"] = "most"
        # save the target\masked image
        print_tensor(target_imgtsr_resized[0], exp_result_root, expriment_meta_data_df, output_type="target_img", **trial_param_dict)

        target_imgtsr_resized_RF_masked =\
            (torch.from_numpy(np.absolute(fitdict_most["fitmap"][None,:,:])) / fitdict_most["fitmap"].max()) *\
            target_imgtsr_resized
        print_tensor(target_imgtsr_resized_RF_masked[0], exp_result_root, expriment_meta_data_df, output_type="target_img_RF_masked", **trial_param_dict)
        
        for score_method in metric_name:
            ## add the stuff to the dictionary
            trial_param_dict["similarity_metric"] = score_method
            for gen_rerun_id in range(NUM_OF_RUNS_PER_IMAGE):
                ## add the generation re-run id to the dictionary
                trial_param_dict["gen_rerun_id"] = gen_rerun_id
                ## set fig title 
                title_str = "%s-%s-%s-popsize%d-most"%(trial_param_dict["trget_imge_name"],
                                        trial_param_dict["similarity_metric"], trial_param_dict["gan_name"],
                                        trial_param_dict["pop_size"])
                #set objective function
                objfunc = set_objective(score_method, targ_actmat, popul_mask=None, normalize=False)
                ## set optimizer
                optimizer = CholeskyCMAES(code_length, population_size=None, init_sigma=3,
                                init_code=np.zeros([1, code_length]), Aupdate_freq=10,
                                maximize=True, random_seed=None, optim_params={})
                
                codes_all, scores_all, actmat_all, generations, last_gem_img, last_gem_img_maske, best_imgs, best_imgs_RF_masked, best_scores =\
                    run_evol(scorer_most, objfunc, optimizer, G, reckey=layer_name, savedir=exp_result_root, meta_data_df=expriment_meta_data_df, trial_param_dict=trial_param_dict,
                    titlestr =title_str, steps=100, RFresize=True, corner=pading_size, imgsize=img_size, RF_mask=fitdict_most["fitmap"], save_plot="full", save_data=True)
                # 
                figh = visualize_popul_act_evol(actmat_all, generations, targ_actmat, titlestr=title_str)
                print_fig(figh, exp_result_root, expriment_meta_data_df, output_type= "popul_act_evol",**trial_param_dict)                                               
            # clear the memory
        scorer_most.cleanup()
        #Let save the meta data frame to the disk in HDF5 format                       
        expriment_meta_data_df.to_hdf(os.path.join(exp_result_root, "expriment_meta_data_df.h5"), key="expriment_meta_data_df", mode="w")
        #%% clear the memory
        torch.cuda.empty_cache() 

                        
