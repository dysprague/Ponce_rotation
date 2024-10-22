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
NUM_OF_RUNS_PER_POP_SIZE = 1
NUM_OF_RUNS_PER_IMAGE = 1
NUM_OF_RUN_RER_SUBPOAP = 1

#%%
# set main fynction
if __name__=="__main__":
    #%%
    #import datetime
    now = datetime.now()
    from core.utils.GAN_utils import upconvGAN
    from core.utils.Optimizers import CholeskyCMAES
    from core.utils.grad_RF_estim import fit_2dgauss, GAN_grad_RF_estimate
    from core.utils.CNN_scorers import TorchScorer, resize_and_pad_tsr
    #%% 
    # Set parameters
    date_time_str = now.strftime("%Y-%m-%d_%H-%M")
    refimgdir = r"C:\Data\cosine\insilico_experiments\papulation_size_effect\data\data_sample_tiny"
    exp_result_root = r"C:\Data\cosine\insilico_experiments\sub_pop_reconstraction\results\tiny_bach_%s" % date_time_str
    os.makedirs(exp_result_root, exist_ok=True)

    net_name = "alexnet"
    layer_name = ".features.Conv2d10"
    layer_short = "conv5"
    gan_name = "fc6"
    pop_size = [2, 4, 8, 16, 32, 64, 128, 256]
    metric_name = ["cosine", "MSE"]
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
                                       "gan_name", "layer_name", "layer_short", "net_name", "img_size", "pading_size", "input_size", "score","pop_unit_idx",
                                       "sub_pop_type","gen_rerun_id", "sub_pop_type", "sub_pop_size", "sub_pop_re_run_id", "save_dir_root", "file_name"])

    #%% save the parameters
    #%% Load the reference images
    refimgnms, refimgtsr = load_ref_imgs(
        imgdir=refimgdir, preprocess_type='center_crop', image_size=227)
    #%% Set up the scorer and the population
    scorer = TorchScorer(net_name)
    module_names, module_types, module_spec = get_module_names(
        scorer.model, input_size, "cuda", False)

    # %% set up the GAN
    G = upconvGAN(gan_name).cuda()
    G.requires_grad_(False)
    code_length = G.codelen

    #%% Run the experiment
    scorer = TorchScorer(net_name)
    for popsize in pop_size:
        #%% add the population size to the dictionary
        trial_param_dict = param_dict.copy()
        trial_param_dict["pop_size"] = popsize
        trial_param_dict["sub_pop_type"] = "whole"
        #%% Set the population
        for pop_resampling_id in range(NUM_OF_RUNS_PER_POP_SIZE):
            # add the population resampling id to the dictionary
            trial_param_dict["pop_resampling_id"] = pop_resampling_id
            # Set scorer and select population    
            
            unit_mask_dict, unit_tsridx_dict = set_random_population_recording(
                scorer, [layer_name], popsize=popsize, seed=pop_resampling_id)
            ## add the population unit index to the dictionary
            trial_param_dict["pop_unit_idx"] = unit_mask_dict[layer_name]
            # Encode a population of images to set the normalizer and mask.
            ref_actmat, _ = encode_image(scorer, refimgtsr, key=layer_name,
                                    RFresize=True, corner=pading_size, imgsize=img_size)
            popul_m, popul_s = set_normalizer(ref_actmat)
            popul_mask = set_popul_mask(ref_actmat)
            # receptive field estimation
            unitslice = (unit_tsridx_dict[layer_name][0],
                        unit_tsridx_dict[layer_name][1][0],
                        unit_tsridx_dict[layer_name][2][0])
            gradAmpmap = GAN_grad_RF_estimate(G, scorer.model, layer_name, unitslice, input_size=input_size,
                                        device="cuda", show=False, reps=100, batch=1)
            fitdict = fit_2dgauss(gradAmpmap)
            #%% Run the evolution
            for imgid in range(len(refimgnms)):
                # Select target image and add target vector.
                targnm, target_imgtsr = refimgnms[imgid], refimgtsr[imgid:imgid + 1]
                targ_actmat, target_imgtsr_resized = encode_image(scorer, target_imgtsr, key=layer_name,
                                        RFresize=True, corner=pading_size, imgsize=img_size)
                targlabel = os.path.splitext(targnm)[0]
                # add the target image name to the dictionary
                trial_param_dict["trget_imge_name"] = targlabel
                #%% save the target\masked image
                print_tensor(target_imgtsr_resized[0], exp_result_root, expriment_meta_data_df, output_type="target_img", **trial_param_dict)

                target_imgtsr_resized_RF_masked =\
                    (torch.from_numpy(np.absolute(fitdict["fitmap"][None,:,:])) / fitdict["fitmap"].max()) *\
                    target_imgtsr_resized
                print_tensor(target_imgtsr_resized_RF_masked[0], exp_result_root, expriment_meta_data_df, output_type="target_img_RF_masked", **trial_param_dict)
                
                for score_method in metric_name:
                    ## add the similarity metric to the dictionary
                    trial_param_dict["similarity_metric"] = score_method
                    for gen_rerun_id in range(NUM_OF_RUNS_PER_IMAGE):
                        ## add the generation re-run id to the dictionary
                        trial_param_dict["gen_rerun_id"] = gen_rerun_id
                        ## set fig title 
                        title_str = "%s-%s-%s-popsize%d"%(trial_param_dict["trget_imge_name"],
                                                trial_param_dict["similarity_metric"], trial_param_dict["gan_name"],
                                                trial_param_dict["pop_size"])
                        #set objective function
                        objfunc = set_objective(score_method, targ_actmat, popul_mask=None, popul_m=popul_m,
                                                popul_s=popul_s)
                        ## set optimizer
                        optimizer = CholeskyCMAES(code_length, population_size=None, init_sigma=3,
                                        init_code=np.zeros([1, code_length]), Aupdate_freq=10,
                                        maximize=True, random_seed=None, optim_params={})
                        
                        codes_all, scores_all, actmat_all, generations, last_gem_img, last_gem_img_maske, best_imgs, best_imgs_RF_masked, best_scores =\
                            run_evol(scorer, objfunc, optimizer, G, reckey=layer_name, savedir=exp_result_root, meta_data_df=expriment_meta_data_df, trial_param_dict=trial_param_dict,
                            titlestr =title_str, steps=100, RFresize=True, corner=pading_size, imgsize=img_size, RF_mask=fitdict["fitmap"], save_plot="full")
                        # 
                        figh = visualize_popul_act_evol(actmat_all, generations, targ_actmat, titlestr=title_str)
                        print_fig(figh, exp_result_root, expriment_meta_data_df, output_type= "popul_act_evol",**trial_param_dict)
                        # sub population analysis
                        # loop ovet the all popolations sizes which is less than popsize/2 and are a power of 2
                        scorer_sub = TorchScorer(net_name)
                        for subpopsize in [2**i for i in range(1, int(np.log2(popsize/2))+1)]:
                            # select sub populations
                            sub_pop_dict = dict()
                            best_rec_actmat = actmat_all[scores_all.argmax()]
                            targ_img_actmat = targ_actmat.squeeze()

                            sub_pop_dict["best"], sub_pop_dict["worst"], sub_pop_dict["random"] = \
                                  select_sub_units(best_rec_actmat, targ_img_actmat, \
                                                   subpopsize, score_method)
                            
                            for sub_pop_type, sub_units in sub_pop_dict.items():
                                # set new dictionary for the sub population
                                trial_param_dict_for_sub = trial_param_dict.copy()
                                trial_param_dict_for_sub["sub_pop_type"] = sub_pop_type
                                trial_param_dict_for_sub["sub_pop_size"] = subpopsize
                                trial_param_dict_for_sub["pop_unit_idx"] = unit_mask_dict[layer_name][sub_units] 
                                title_str_sub = f"{title_str}-{sub_pop_type}-{subpopsize}"
                                
                                flat_idx_samp = unit_mask_dict[layer_name][sub_units] 
                                scorer_sub.set_popul_recording(layer_name, flat_idx_samp, )

                                #%% receptive field estimation
                                unitslice_sub = (unit_tsridx_dict[layer_name][0][sub_units],
                                                unit_tsridx_dict[layer_name][1][0],
                                                unit_tsridx_dict[layer_name][2][0])
                                gradAmpmap_sub = GAN_grad_RF_estimate(G, scorer_sub.model, layer_name, unitslice, input_size= input_size,
                                                                device="cuda", show=False, reps=100, batch=1)
                                fitdict_sub = fit_2dgauss(gradAmpmap_sub)

                                targ_actmat_sub, _ = encode_image(scorer_sub, target_imgtsr, key=layer_name,
                                            RFresize=True, corner=pading_size, imgsize=img_size)

                                objfunc_sub = set_objective(score_method, targ_actmat_sub, popul_mask=None, popul_m=popul_m[:, sub_units],
                                                popul_s=popul_s[:, sub_units])

                                optimizer_sub = CholeskyCMAES(code_length, population_size=None, init_sigma=3,
                                        init_code=np.zeros([1, code_length]), Aupdate_freq=10,
                                        maximize=True, random_seed=None, optim_params={})
                                        
                                codes_all_sub, scores_all_sub, actmat_all_sub, generations_sub, last_gem_img_sub,\
                                last_gem_img_maske_sub, best_imgs_sub, best_imgs_RF_masked_sub, best_scores_sub =\
                                    run_evol(scorer_sub, objfunc_sub, optimizer_sub, G, reckey=layer_name, savedir=exp_result_root,
                                            meta_data_df=expriment_meta_data_df, trial_param_dict=trial_param_dict_for_sub,
                                             titlestr=title_str_sub, steps=100, RFresize=True, corner=pading_size, imgsize=img_size, RF_mask=fitdict_sub["fitmap"], save_plot="full")
                                
                                figh = visualize_popul_act_evol(actmat_all_sub, generations_sub, targ_actmat_sub, titlestr=title_str_sub)
                                print_fig(figh, exp_result_root, expriment_meta_data_df, output_type= "popul_act_evol",**trial_param_dict_for_sub)
                                plt.close("all")
                                scorer_sub.cleanup()
                                #del optimizer_sub, objfunc_sub, gradAmpmap_sub, fitdict_sub, targ_actmat_sub, trial_param_dict_for_sub        
                                torch.cuda.empty_cache()                       
                                #Let save the meta data frame to the disk in HDF5 format
                                expriment_meta_data_df.to_hdf(os.path.join(exp_result_root, "expriment_meta_data_df.h5"), key="expriment_meta_data_df", mode="w")
            scorer.cleanup()
            #del optimizer, objfunc, gradAmpmap, fitdict, targ_actmat, trial_param_dict
            torch.cuda.empty_cache() 

                        




