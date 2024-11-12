"""
Library of functions useful for Recording population response and do Cosine Evolution.
"""
import time
from os.path import join
import matplotlib.pylab as plt
import os
import torch
import numpy as np
import copy
import scipy
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize, ToPILImage, CenterCrop
from core.utils.CNN_scorers import TorchScorer, visualize_trajectory, resize_and_pad_tsr
from core.utils.layer_hook_utils import get_module_names, get_layer_names
from core.utils.grad_RF_estim import fit_2dgauss, GAN_grad_RF_estimate

#from core.utils.CNN_scorers import TorchScorer

def run_evol(scorer, objfunc, optimizer, G, reckey=None, steps=100, savedir=None, meta_data_df=None, trial_param_dict=None,
            RFresize=True, corner=(0, 0), imgsize=(224, 224), init_code=None, RF_mask=None, save_plot="brief", save_data=False,
            titlestr="", printsteps=False):
    if meta_data_df is None:
        Warning ("meta_data_df is None, will not save the results")
    if savedir is None:
        savedir = os.getcwd()  # current working directory
    if init_code is None:
        init_code = np.zeros((1, G.codelen))
    new_codes = init_code
    # new_codes = init_code + np.random.randn(25, 256) * 0.06
    scores_all = []
    actmat_all = []
    generations = []
    codes_all = []
    best_codes = []
    best_imgs = []
    best_scores = []
    init_score = []
    init_score_flag = True
    for i in range(steps,):
        codes_all.append(new_codes.copy())
        T0 = time.time()  #process_
        imgs = G.visualize_batch_np(new_codes)  # B=1
        latent_code = torch.from_numpy(np.array(new_codes)).float()
        T1 = time.time()  #process_
        if RFresize: imgs = resize_and_pad_tsr(imgs, imgsize, corner, )
        T2 = time.time()  #process_
        _, recordings = scorer.score_tsr(imgs)        
        if isinstance(reckey, list):
            actmat = np.concatenate([recordings[layer_n] for layer_n in reckey], axis=1) 
        else:
            actmat = recordings[reckey]
        T3 = time.time()  #process_
        scores = objfunc(actmat, imgs)  # targ_actmat
        if init_score_flag:
            init_score_flag = False
            init_score.append(scores.mean())

        T4 = time.time()  #process_
        new_codes = optimizer.step_simple(scores, new_codes, )
        T5 = time.time()  #process_
        # print the results of each step
        if printsteps:
            if "BigGAN" in str(G.__class__):
                print("step %d score %.3f (%.3f) (norm %.2f noise norm %.2f)" % (
                    i, scores.mean(), scores.std(), latent_code[:, 128:].norm(dim=1).mean(),
                    latent_code[:, :128].norm(dim=1).mean()))
            else:
                print("step %d score %.3f (%.3f) (norm %.2f )" % (
                    i, scores.mean(), scores.std(), latent_code.norm(dim=1).mean(),))
            print(f"GANvis {T1-T0:.3f} RFresize {T2-T1:.3f} CNNforw {T3-T2:.3f}  "
                f"objfunc {T4-T3:.3f}  optim {T5-T4:.3f} total {T5-T0:.3f}")
            
        scores_all.extend(list(scores))
        generations.extend([i] * len(scores))
        best_imgs.append(imgs[scores.argmax(),:,:,:].detach().clone())
        best_scores.append(scores.max())
        best_codes.append(new_codes[scores.argmax(),:].copy())
        # debug @ jan.3rd. Before there is serious memory leak `.detach().clone()` solve the reference issue.
        actmat_all.append(actmat)
    codes_all = np.concatenate(tuple(codes_all), axis=0)
    best_codes = np.vstack(best_codes)
    scores_all = np.array(scores_all)
    actmat_all = np.concatenate(tuple(actmat_all), axis=0)
    generations = np.array(generations)
    mtg_exp = ToPILImage()(make_grid(best_imgs, nrow=10))
    mtg = ToPILImage()(make_grid(imgs, nrow=7))
    lastgen_mean = torch.mean(imgs, dim=0)
    # masking
    best_imgs_RF_masked = None
    last_gen_masked = None
    mtg_exp_RF_masked = None
    mtg_RF_masked = None
    last_gen_masked_mean = None

    if RF_mask is not None:
        best_imgs_RF_masked = []
        for i in range(len(best_imgs)):
            best_imgs_RF_masked.append((torch.from_numpy(np.absolute(RF_mask[None,:,:])) / RF_mask.max()) * best_imgs[i])
        #from core.utils.plot_utils import show_imgrid
        #show_imgrid(mtg_exp_RF_masked)
        mtg_exp_RF_masked = ToPILImage()(make_grid(best_imgs_RF_masked, nrow=10))
        

        last_gen_masked = (torch.from_numpy(np.absolute(RF_mask[None,None,:,:])) / RF_mask.max()) * imgs
        mtg_RF_masked = ToPILImage()(make_grid(last_gen_masked, nrow=7))
        last_gen_masked_mean = torch.mean(last_gen_masked, dim=0)

    print(":-------------------->  %s evol finished! best score: %.3f"%(titlestr, np.max(best_scores)))            
    # save the results
    if save_data:
        #TODO: NEED TO FIX THIS PART---> mabey you want to save more image code. consider for fc6 you cant save all
        data_dict = dict(generations=generations,
                scores_all=scores_all, actmat_all=actmat_all, RF_mask=RF_mask, best_codes=best_codes)
        data_saver(data_dict, savedir, meta_data_df, output_type="generation_data",
                score=np.max(best_scores),init_score= init_score, **trial_param_dict)      
    
    # plot the results    
    if save_plot == "brief":
        print_tensor(best_imgs[np.argmax(best_scores)], savedir, meta_data_df, output_type="best_gen_imgs",
                    score=np.max(best_scores),init_score= init_score,**trial_param_dict)
        print_tensor(best_imgs_RF_masked[np.argmax(best_scores)], savedir, meta_data_df, output_type="best_gen_imgs_RF_masked",
                     score=np.max(best_scores),init_score= init_score, **trial_param_dict)
        trajectory_fig = visualize_trajectory(scores_all, generations, codes_arr=codes_all,
                                                title_str=titlestr)
        print_fig(trajectory_fig, savedir, meta_data_df,output_type= "optimization_trajectory", score=np.mean(best_scores),init_score= init_score, **trial_param_dict)
        
    elif save_plot == "full":

        print_tensor(best_imgs[np.argmax(best_scores)], savedir, meta_data_df, output_type="best_gen_imgs",
                    score=np.max(best_scores),init_score= init_score,**trial_param_dict)
        print_tensor(best_imgs_RF_masked[np.argmax(best_scores)], savedir, meta_data_df, output_type="best_gen_imgs_RF_masked",
                     score=np.max(best_scores),init_score= init_score, **trial_param_dict)
        
        print_tensor(lastgen_mean, savedir, meta_data_df, output_type= "last_gen_mean_imgs",
                     score=np.mean(best_scores),init_score= init_score, **trial_param_dict)
        print_tensor(last_gen_masked_mean, savedir, meta_data_df, output_type= "last_gen_mean_imgs_RF_masked",
                     score=np.mean(best_scores),init_score= init_score, **trial_param_dict)
        
        print_tensor(mtg, savedir, meta_data_df, PILimage=True,output_type= "last_gen_all_imgs", score=np.mean(best_scores),init_score= init_score, **trial_param_dict)
        print_tensor(mtg_exp, savedir, meta_data_df, PILimage=True,output_type= "best_in_each_gen_imgs",  **trial_param_dict)
        
        print_tensor(mtg_RF_masked, savedir, meta_data_df, PILimage=True,output_type= "last_gen_all_imgs_RF_masked", score=np.mean(best_scores),init_score= init_score, **trial_param_dict)
        print_tensor(mtg_exp_RF_masked, savedir, meta_data_df, PILimage=True,output_type= "best_in_each_gen_imgs_RF_masked",  **trial_param_dict)

        trajectory_fig = visualize_trajectory(scores_all, generations, codes_arr=codes_all,
                                                title_str=titlestr)
        print_fig(trajectory_fig, savedir, meta_data_df,output_type= "optimization_trajectory", score=np.mean(best_scores), init_score= init_score,**trial_param_dict)
    elif save_plot == None:
        print("plot not saved")
    else:
        raise NotImplementedError("save_plot should be one of 'brief', 'full', 'none'")
    
    plt.close('all') 
    return codes_all, scores_all, actmat_all, generations, imgs, last_gen_masked, best_imgs, best_imgs_RF_masked, best_scores, best_codes
#%%
def sample_center_units_idx(tsrshape, samplenum=500, single_col=True, area=False, resample=False):
    """

    :param tsrshape: shape of the tensor to be sampled
    :param samplenum: total number of unit to sample
    :param single_col: restrict the sampling to be from a single column
    :param resample: allow the same unit to be sample multiple times or not?
    :return:
        flat_idx_samp: a integer array to sample the flattened feature tensor
    """
    msk = np.zeros(tsrshape, dtype=np.bool) # the viable units in the center of the featuer map
    if len(tsrshape)==3:
        C, H, W = msk.shape
        if single_col: # a single column
            msk[:, int(H//2), int(W//2)] = True
        else: 
            if area: # a area in the center
                msk[:,
                    int(H/4):int(3*H/4),
                    int(W/4):int(3*W/4)] = True
            else: # all units
                msk[:] = True
    else:
        msk[:] = True
    center_idxs = np.where(msk.flatten())[0]
    flat_idx_samp = np.random.choice(center_idxs, samplenum, replace=resample)
    flat_idx_samp.sort()
    #     np.unravel_index(flat_idx_samp, outshape)
    return flat_idx_samp

def sample_center_column_units_idx(tsrshape, single_col=True):
    """ Return index of center column or the center columns.

    :param tsrshape: shape of the tensor to be sampled
    :param single_col: restrict the sampling to be from a single column
    :return:
        flat_idx_samp: a integer array to sample the flattened feature tensor
    """
    msk = np.zeros(tsrshape, dtype=np.bool) # the viable units in the center of the featuer map
    if len(tsrshape) == 3:
        C, H, W = msk.shape
        if single_col: # a single column
            msk[:, int(H//2), int(W//2)] = True
        else: # a area in the center
            msk[:,
                int(H/4):int(3*H/4),
                int(W/4):int(3*W/4)] = True
    else:
        msk[:] = True
    center_idxs = np.where(msk.flatten())[0]
    center_idxs.sort()
    return center_idxs


def set_random_population_recording(scorer, targetnames, randomize=True, popsize=500, single_col=True, resample=False,
                                    seed=None, print_info=True, area=False):
    """ Main effect is to set the recordings for the scorer object.
    (additional method for scorer)

    :param scorer:
    :param targetnames:
    :param popsize:
    :param single_col: restrict the sampling to be from a single column
    :param resample: allow the same unit to be sample multiple times or not?
    :return:

    """
    np.random.seed(seed) # set a seed for reproducing population selection
    unit_mask_dict = {}
    unit_tsridx_dict = {}
    module_names, module_types, module_spec = get_module_names(scorer.model, (3,227,227), "cuda", False)
    invmap = {v: k for k, v in module_names.items()}
    try:
        for layer in targetnames:
            inshape = module_spec[invmap[layer]]["inshape"]
            outshape = module_spec[invmap[layer]]["outshape"]
            if randomize:
                flat_idx_samp = sample_center_units_idx(outshape, popsize, single_col=single_col, resample=resample, area=area)
            else:
                flat_idx_samp = sample_center_column_units_idx(outshape, single_col=True)
                popsize = len(flat_idx_samp)

            tsr_idx_samp = np.unravel_index(flat_idx_samp, outshape)
            unit_mask_dict[layer] = flat_idx_samp
            unit_tsridx_dict[layer] = tsr_idx_samp
            scorer.set_popul_recording(layer, flat_idx_samp, )
            if print_info: print(f"Layer {layer} Sampled {popsize} units from feature tensor of shape {outshape}")
    except KeyError:
        print(*invmap.keys(), sep="\n")
        raise KeyError
    return unit_mask_dict, unit_tsridx_dict

def set_all_center_unit_population_recording(scorer, targetnames, print_info=True):
    """ Main effect is to set the recordings for the scorer object.
    """
    unit_mask_dict = {}
    unit_tsridx_dict = {}
    module_names, module_types, module_spec = get_module_names(scorer.model, (3,227,227), "cuda", False)
    invmap = {v: k for k, v in module_names.items()}
    print(module_spec)
    print(invmap)
    print(targetnames)
    try:
        for layer in targetnames:
            inshape = module_spec[invmap[layer]]["inshape"]
            outshape = module_spec[invmap[layer]]["outshape"]          
            flat_idx_samp = sample_center_column_units_idx(outshape, single_col=True)
            tsr_idx_samp = np.unravel_index(flat_idx_samp, outshape)
            unit_mask_dict[layer] = flat_idx_samp
            unit_tsridx_dict[layer] = tsr_idx_samp
            scorer.set_popul_recording(layer, flat_idx_samp, )
            if print_info: print(f"Layer {layer} center units from feature tensor of shape {outshape}")
    except KeyError:
        print(*invmap.keys(), sep="\n")
        raise KeyError
    return unit_mask_dict, unit_tsridx_dict

def set_all_unit_population_recording(scorer, targetnames, print_info=True):
    """ Main effect is to set the recordings for the scorer object.
    """
    unit_mask_dict = {}
    unit_tsridx_dict = {}
    if torch.cuda.is_available():
        module_names, module_types, module_spec = get_module_names(scorer.model, (3,227,227), "cuda", False)
    else:
        module_names, module_types, module_spec = get_module_names(scorer.model, (3,227,227), "cpu", False)
    invmap = {v: k for k, v in module_names.items()}

    print(module_spec)
    print(invmap)
    print(targetnames)
    try:
        for layer in targetnames:
            print(invmap[layer])
            inshape = module_spec[invmap[layer]]["inshape"]
            outshape = module_spec[invmap[layer]]["outshape"]   
            msk = np.ones(outshape, dtype=np.bool)  
            flat_idx_samp = np.where(msk.flatten())[0]
            flat_idx_samp.sort()     

            tsr_idx_samp = np.unravel_index(flat_idx_samp, outshape)
            unit_mask_dict[layer] = flat_idx_samp
            unit_tsridx_dict[layer] = tsr_idx_samp
            scorer.set_popul_recording(layer, flat_idx_samp, )
            if print_info: print(f"Layer {layer} all units from feature tensor of shape {outshape}")
    except KeyError:
        print(*invmap.keys(), sep="\n")
        raise KeyError
    return unit_mask_dict, unit_tsridx_dict
#%%
def set_most_active_population_recording(scorer_most, targetnames, refimgtsr, corner_pading, img_size, popsize=500, single_col=True):
    most_active_unit_tsridx_dict = {}
    most_active_unit_mask_dict = {}
    module_names, module_types, module_spec = get_module_names(scorer_most.model, (3,227,227), "cuda", False)
    invmap = {v: k for k, v in module_names.items()}
    try:
        # set empty dict
        most_actve_unit_idx = {}
        flat_idx_most_actve = {}
        for layer in targetnames:
            outshape = module_spec[invmap[layer]]["outshape"]
            scorer_most_finder = copy.deepcopy(scorer_most)
            if single_col:
                flat_idx_all = sample_center_column_units_idx(outshape, single_col=True)
            else:
                msk = np.zeros(outshape, dtype=np.bool)
                center_idxs = np.where(msk.flatten())[0]
                flat_idx_all = center_idxs.sort()
            scorer_most_finder.set_popul_recording(layer, flat_idx_all, )
            ref_actmat, _ = encode_image(scorer_most_finder, refimgtsr, key=layer,
                            RFresize=True, corner=corner_pading, imgsize=img_size)   
            # get popsiz elements largest values of ref_actmat
            most_actve_unit_idx[layer] = np.argsort(ref_actmat, axis=None)[-popsize:] 
            flat_idx_most_actve[layer] = flat_idx_all[most_actve_unit_idx[layer]]

            scorer_most_finder.cleanup(print_info=False)

        for layer in targetnames:
            outshape = module_spec[invmap[layer]]["outshape"]
            tsr_idx_most_active = np.unravel_index(flat_idx_most_actve[layer], outshape)
            most_active_unit_mask_dict[layer] = flat_idx_most_actve[layer]
            most_active_unit_tsridx_dict[layer] = tsr_idx_most_active
            scorer_most.set_popul_recording(layer, flat_idx_most_actve[layer], )

    except KeyError:
        print(*invmap.keys(), sep="\n")
        raise KeyError
    return most_active_unit_mask_dict, most_active_unit_tsridx_dict


def set_least_active_population_recording(scorer_least, targetnames, refimgtsr, corner_pading, img_size, popsize=500, single_col=True):
    least_active_unit_tsridx_dict = {}
    least_active_unit_mask_dict = {}
    module_names, module_types, module_spec = get_module_names(scorer_least.model, (3,227,227), "cuda", False)
    invmap = {v: k for k, v in module_names.items()}
    try:
        for layer in targetnames:
            outshape = module_spec[invmap[layer]]["outshape"]

            flat_idx_all = sample_center_column_units_idx(outshape, single_col=True)
            scorer_least.set_popul_recording(layer, flat_idx_all, )
            ref_actmat, _ = encode_image(scorer_least, refimgtsr, key=layer,
                            RFresize=True, corner=corner_pading, imgsize=img_size)   
            # get popsiz elements smallest values of ref_actmat
            least_actve_unit_idx = np.argsort(ref_actmat, axis=None)[:popsize]
            
            flat_idx_least_actve = flat_idx_all[least_actve_unit_idx]
            
            scorer_least.cleanup(print_info=False)
        
            tsr_idx_most_active = np.unravel_index(flat_idx_least_actve, outshape)
            least_active_unit_mask_dict[layer] = flat_idx_least_actve
            least_active_unit_tsridx_dict[layer] = tsr_idx_most_active
            scorer_least.set_popul_recording(layer, flat_idx_least_actve, )

    except KeyError:
        print(*invmap.keys(), sep="\n")
        raise KeyError
    return least_active_unit_mask_dict, least_active_unit_tsridx_dict
  
def set_objective(score_method, targmat, popul_mask=None, popul_m=None, popul_s=None,
                  normalize=True, VI=None, unit_id=None, alpa_reg_landa = .005,  tv_req_lambda = 0.000000005, **kwargs ):
    if popul_mask is None:
        popul_mask = slice(None)
    def objfunc(actmat, img_tsr):
        assert img_tsr.ndim in [3, 4]
        if img_tsr.ndim == 3:
            img_tsr.unsqueeze_(0)

        actmat_msk = actmat[:, popul_mask]
        targmat_msk = targmat[:, popul_mask] # [1 by masksize]
        if normalize:
            actmat_msk = (actmat_msk - popul_m[:, popul_mask]) / popul_s[:, popul_mask]
            targmat_msk = (targmat_msk - popul_m[:, popul_mask]) / popul_s[:, popul_mask]
        if score_method == "L1":
            scores = - np.abs(actmat_msk - targmat_msk).mean(axis=1)
        elif score_method == "MSE":
            scores = - np.square(actmat_msk - targmat_msk).mean(axis=1)
        elif score_method == "MSE_normalized":
            scores = - np.square((actmat_msk/np.linalg.norm(actmat_msk, axis=1)[:, None]) - 
                                 (targmat_msk/np.linalg.norm(targmat_msk, axis=1)[:, None])).mean(axis=1)
        elif score_method == "Chebyshev":
            scores = - np.max(np.abs(actmat_msk - targmat_msk), axis=1)
        elif score_method == "Correlation":
            scores = np.empty(actmat_msk.shape[0])
            for i in range(actmat_msk.shape[0]):
                scores[i] = np.corrcoef(actmat_msk[i, :], targmat_msk[0, :])[0, 1]
        elif score_method == "Spearman":
            scores = np.empty(actmat_msk.shape[0])
            for i in range(actmat_msk.shape[0]):
                scores[i] = scipy.stats.spearmanr(actmat_msk[i, :], targmat_msk[0, :], axis=1).correlation
        elif score_method == "corr":
            actmat_msk = actmat_msk - actmat_msk.mean() # there is a bug right? actmat_msk - actmat_msk.mean(axis=1, keepdims=True)
            targmat_msk = targmat_msk - targmat_msk.mean()
            popact_norm = np.linalg.norm(actmat_msk, axis=1, keepdims=True)
            targact_norm = np.linalg.norm(targmat_msk, axis=1, keepdims=True)
            scores = ((actmat_msk @ targmat_msk.T) / popact_norm / targact_norm).squeeze(axis=1)
        elif score_method == "cosine":
            popact_norm = np.linalg.norm(actmat_msk, axis=1,keepdims=True)
            targact_norm = np.linalg.norm(targmat_msk, axis=1,keepdims=True)
            scores = ((actmat_msk @ targmat_msk.T) / popact_norm / targact_norm).squeeze(axis=1)
        elif score_method == "cosine_norm_diff":
            popact_norm = np.linalg.norm(actmat_msk, axis=1,keepdims=True)
            targact_norm = np.linalg.norm(targmat_msk, axis=1,keepdims=True)
            scores_cosine = ((actmat_msk @ targmat_msk.T) / popact_norm / targact_norm).squeeze(axis=1)
            scores_norm_diff = np.abs(np.linalg.norm(targmat_msk, axis=1) - np.linalg.norm(actmat_msk, axis=1)) / (np.linalg.norm(targmat_msk, axis=1) + np.linalg.norm(actmat_msk, axis=1))
            scores = scores_cosine - scores_norm_diff
        elif score_method == "cosine_req":
            popact_norm = np.linalg.norm(actmat_msk, axis=1,keepdims=True)
            targact_norm = np.linalg.norm(targmat_msk, axis=1,keepdims=True)
            cosine_score = ((actmat_msk @ targmat_msk.T) / popact_norm / targact_norm).squeeze(axis=1)

            # add two reqularization terms to the cosine similarity, (1) a total variation (TV) regularizer, and (2) an ɑ-norm regularizer.
            # TV regularizer: penalize the difference between adjacent units
            # ɑ-norm regularizer: penalize the difference between the activation of each unit and the average activation of the population
            # ɑ is a hyperparameter that controls the strength of the regularization
            # TV regularizer
            img_tsr_sub = img_tsr-img_tsr.mean()
            # TV regularizer
            tv_reg = (torch.sum(
                torch.abs(img_tsr_sub[:, :, :, :-1] - img_tsr_sub[:, :, :, 1:]), dim = (1,2,3)) + \
                torch.sum(torch.abs(img_tsr_sub[:, :, :-1, :] - img_tsr_sub[:, :, 1:, :]), dim = (1,2,3))).numpy()
            tv_req_lambda_val = tv_req_lambda # you may Xval over this hyperparameter
            # ɑ-norm regularizer
            alpa_reg_landa_val = alpa_reg_landa # you may Xval over this hyperparameter
            alpha_val = 6
            alpha_req = torch.norm(img_tsr_sub, p=alpha_val, dim=(1,2,3)).numpy()

            scores = cosine_score - (tv_req_lambda_val*tv_reg) - (alpa_reg_landa_val*alpha_req)
            
        elif score_method == "dot":
            scores = (actmat_msk @ targmat_msk.T).squeeze(axis=1)
        elif score_method == "euclidean":
            
            targact_norm = np.linalg.norm(targmat_msk, axis=1, keepdims=True)
            diff_norm = np.linalg.norm((actmat_msk - targmat_msk), axis=1, keepdims=True)
            scores = - (diff_norm/targact_norm).squeeze(axis=1)
        elif score_method == "euclidean_req":
            # add two reqularization terms to the euclidean distance, (1) a total variation (TV) regularizer, and (2) an ɑ-norm regularizer.
            # TV regularizer: penalize the difference between adjacent units
            # ɑ-norm regularizer: penalize the difference between the activation of each unit and the average activation of the population
            # ɑ is a int hyperparameter that controls the strength of the regularization
            # ref: https://arxiv.org/abs/1412.0035 
            # ref: https://www.biorxiv.org/content/10.1101/2022.05.19.492678v2
            targact_norm = np.linalg.norm(targmat_msk, axis=1, keepdims=True)
            diff_norm = np.linalg.norm((actmat_msk - targmat_msk), axis=1, keepdims=True)
            euclidean_scores = (diff_norm/targact_norm).squeeze(axis=1)
            
            img_tsr_sub = img_tsr-img_tsr.mean()
            # TV regularizer
            tv_reg = (torch.sum(
                torch.abs(img_tsr_sub[:, :, :, :-1] - img_tsr_sub[:, :, :, 1:]), dim = (1,2,3)) + \
                torch.sum(torch.abs(img_tsr_sub[:, :, :-1, :] - img_tsr_sub[:, :, 1:, :]), dim = (1,2,3))).numpy()
            tv_req_lambda_val = tv_req_lambda # you may Xval over this hyperparameter
            # ɑ-norm regularizer
            alpa_reg_landa_val = alpa_reg_landa # you may Xval over this hyperparameter
            alpha_val = 6
            alpha_req = torch.norm(img_tsr_sub, p=alpha_val, dim=(1,2,3)).numpy()

            scores = - (euclidean_scores + (alpa_reg_landa_val * alpha_req) + (tv_req_lambda_val * tv_reg))
        elif score_method == "Mahalanobis":
      #      print("I''m in Mahalanobis")
            if VI is None:
                raise ValueError("Mahalanobis distance requires a covariance matrix VI")
            scores = np.empty(actmat_msk.shape[0])
            for i in range(actmat_msk.shape[0]):
                scores[i] = -1* scipy.spatial.distance.mahalanobis(actmat_msk[i, :], targmat_msk[0, :], VI)
        elif score_method == "unit_killing":
            # scores = targmat_msk[:, unit_id] - actmat_msk[:, unit_id]

            if  np.mean(actmat_msk[:, unit_id]) > 0: #TODO: consider bias
                scores = - actmat_msk[:, unit_id] 
            else:
                scores = actmat_msk[:, unit_id] 
        elif score_method == "inflate_unit":
            scores = actmat_msk[:, unit_id]
        else:
            raise ValueError
        return scores # (Nimg, ) 1d array
    # return an array / tensor of scores for an array of activations
    # Noise form
    return objfunc

def set_objective_single_unit(score_method, inital_actmat, unit_id, popul_mask=None, popul_m=None, popul_s=None,
                  normalize=True, **kwargs ):
    if popul_mask is None:
        popul_mask = slice(None)
    def objfunc(actmat, img_tsr):
        assert img_tsr.ndim in [3, 4]
        if img_tsr.ndim == 3:
            img_tsr.unsqueeze_(0)

        actmat_msk = actmat[:, popul_mask]
        inital_actmat_msk = inital_actmat[:, popul_mask] # [1 by masksize]
        if normalize:
            actmat_msk = (actmat_msk - popul_m[:, popul_mask]) / popul_s[:, popul_mask]
            inital_actmat_msk = (inital_actmat_msk - popul_m[:, popul_mask]) / popul_s[:, popul_mask]
        if score_method == "unit_killing":
            scores = inital_actmat_msk[:, unit_id] - actmat_msk[:, unit_id] # TODO: you can make the cell zerow (considering bias)
            
        else:
            raise ValueError
        return scores # (Nimg, ) 1d array
    # return an array / tensor of scores for an array of activations
    # Noise form
    return objfunc

def set_objective_grad(score_method, targmat, popul_mask, popul_m, popul_s,
                       normalize=True, device="cuda"):
    """ PyTorch version of the objective function, suppoorting gradient flow.
    translated from numpy version above.
    :param score_method: str, one of "L1", "MSE", "corr", "cosine", "dot"
    :param targmat: (1, Nfeat) torch tensor
    :param popul_mask: (Nfeat,) torch tensor or None or slice.
                        If None, then all features are used
    :param popul_m: None or (1, Nfeat) torch tensor,
                mean of activation used for normalization
    :param popul_s: None or (1, Nfeat) torch tensor,
                std of activation used for normalization
    :param normalize: bool, whether to normalize the activations
                if False, then the popul_m and popul_s are ignored
    :return:
    """
    if popul_mask is None:
        popul_mask = slice(None)
    targmat = targmat.to(device)
    popul_m = popul_m.to(device) if popul_m is not None else None
    popul_s = popul_s.to(device) if popul_s is not None else None
    def objfunc(actmat):
        actmat_msk = actmat[:, popul_mask].to(device)
        targmat_msk = targmat[:, popul_mask]  # [1 by masksize]
        if normalize:
            actmat_msk = (actmat_msk - popul_m[:, popul_mask]) / popul_s[:, popul_mask]
            targmat_msk = (targmat_msk - popul_m[:, popul_mask]) / popul_s[:, popul_mask]

        if score_method == "L1":
            scores = - (actmat_msk - targmat_msk).abs().mean(dim=1)
        elif score_method == "MSE":
            scores = - (actmat_msk - targmat_msk).pow(2).mean(dim=1)
        elif score_method == "corr":
            actmat_msk_sub = actmat_msk - actmat_msk.mean(dim=1, keepdim=True)  # there is a bug right? actmat_msk - actmat_msk.mean(axis=1, keepdims=True)
            targmat_msk_sub= targmat_msk - targmat_msk.mean()
            popact_norm = actmat_msk_sub.norm(dim=1, keepdim=True)
            targact_norm = targmat_msk_sub.norm(dim=1, keepdim=True)
            scores = ((actmat_msk_sub @ targmat_msk_sub.T) / popact_norm / targact_norm).squeeze(dim=1)
        elif score_method == "cosine":
            popact_norm = actmat_msk.norm(dim=1, keepdim=True)
            targact_norm = targmat_msk.norm(dim=1, keepdim=True)
            scores = ((actmat_msk @ targmat_msk.T) / popact_norm / targact_norm).squeeze(dim=1)
        elif score_method == "dot":
            scores = (actmat_msk @ targmat_msk.T).squeeze(dim=1)
        else:
            raise ValueError(" score method not recognized - may you want to add it?") 
        return scores  # (Nimg, ) 1d array

    # return an array / tensor of scores for an array of activations
    # Noise form
    return objfunc


def encode_image(scorer, imgtsr, key=None,
                 RFresize=True, corner=None, imgsize=None, cat_layes=True):
    """return a 2d array / tensor of activations for a image tensor
    imgtsr: (Nimgs, C, H, W)
    actmat: (Npop, Nimages) torch tensor

    :return
        if key is None then return a dict of all actmat of all layer
        if key is in the dict, then return a single actmat of shape (imageN, unitN)
    """
    #TODO: make this work for larger image dataset
    _, recordings = scorer.score_tsr(imgtsr)
    if RFresize: imgtsr = resize_and_pad_tsr(imgtsr, imgsize, corner, )
    _, recordings = scorer.score_tsr(imgtsr)
    if key is None:
        return recordings, imgtsr
    else:
        if isinstance(key, list):
            actmat = [recordings[layer_n] for layer_n in key]
            if cat_layes:
                actmat = np.concatenate(actmat, axis=1)
            return actmat, imgtsr
        else:
            return recordings[key], imgtsr
        

def set_popul_mask(ref_actmat):
    img_var = ref_actmat.var(axis=0) # (unitN, )
    popul_mask = ~np.isclose(img_var, 0.0)
    # if these inactive units are not excluded, then the normalization will be nan.
    print(popul_mask.sum(), " units still active in the mask.")
    return popul_mask


def set_normalizer(ref_actmat):
    """ Get normalizer for activation. by default return mean and std.

    :param ref_actmat: torch tensor of shape (imageN, unitN)
    :return:
    """
    return ref_actmat.mean(axis=0, keepdims=True), ref_actmat.std(axis=0, keepdims=True),



from cycler import cycler
from matplotlib.cm import jet
def visualize_popul_act_evol(actmat_all, generations, targ_actmat, titlestr=""):
    """
    # figh = visualize_popul_act_evol(actmat_all, generations, targ_actmat)
    # figh.savefig(join(expdir, "popul_act_evol_%s_%d.png" % (explabel, RND)))

    :param actmat_all:
    :param generations:
    :param targ_actmat:
    :return:
    """
    Ngen = generations.max() + 1
    actmat_tr_avg = np.array([actmat_all[generations==gi, :].mean(axis=0) for gi in range(Ngen)])
    sortidx = targ_actmat.argsort()[0]
    figh= plt.figure(figsize=[10, 8])
    ax = plt.gca()
    ax.set_prop_cycle(cycler(color=[jet(k) for k in np.linspace(0,1,Ngen)]))
    plt.plot(actmat_tr_avg[:,sortidx].T, alpha=0.3, lw=1.5)
    plt.plot(targ_actmat[:,sortidx].T, color='k', alpha=0.8,lw=2.5)
    plt.xlabel("populatiion unit (sorted by target pattern)")
    plt.ylabel("activation")
    plt.title("Neural Pattern Evolution\n %s" % titlestr)
    plt.tight_layout()
    # plt.show()
    return figh

def visualize_single_cell_modification(actmat_all, generations, unit_idx, titlestr=""):
    """
    # figh = visualize_popul_act_evol(actmat_all, generations, targ_actmat)
    # figh.savefig(join(expdir, "popul_act_evol_%s_%d.png" % (explabel, RND)))

    :param actmat_all:
    :param generations:
    :param targ_actmat:
    :return:
    """
    Ngen = generations.max() + 1
    cell_act_tr_avg = np.array([actmat_all[generations==gi, unit_idx].mean() for gi in range(Ngen)])
    cell_act_tr = actmat_all[:, unit_idx]
    figh = plt.figure(figsize=[10, 8])
    plt.scatter(generations, cell_act_tr, alpha=0.3, s=5, c='gray')
    plt.plot(range(len(cell_act_tr_avg)), cell_act_tr_avg, alpha=0.7, lw=4, c='k')
    plt.xlabel("gen id", fontsize=20)
    plt.ylabel("activation of unit", fontsize=20)
    plt.title(f"unit {unit_idx} modification\n {titlestr}", fontsize=20)
    plt.tight_layout()
    # plt.show()
    return figh

def load_ref_imgs(imgdir, preprocess_type = "center_crop", image_size = 224, Nlimit=200):
    
    if not(isinstance(image_size, int)):
        raise ValueError("image_size should be a single int number")

    # set up preprocessing
    if preprocess_type == "center_crop":
        preprocess = Compose([           
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor()])
    elif preprocess_type == "resize":
        preprocess = Compose([
            Resize((image_size, image_size)),
            ToTensor()])
    else:
        raise ValueError("preprocess_type not recognized - my you want to add it?")
    
    imgs = []
    imgnms = []
    valid_images = [".jpg", ".gif", ".png", ".tga", ".jpeg", ".bmp", ".tif"]
    for f in os.listdir(imgdir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(preprocess(Image.open(os.path.join(imgdir, f)).convert('RGB')))
        imgnms.append(f)
        if Nlimit is not None:
            if len(imgs) > Nlimit:
                break
                Warning("Nlimit reached, only %d images loaded" % Nlimit)
    imgtsr = torch.stack(imgs)
    return imgnms, imgtsr

def get_image_distance_tensor(targer_image, source_images, distance_metric='L2'):
    """Compute distance between a target image and a set of source images
    :param targer_image: 3d tensor of shape (C, H, W)
    :param source_images: 4d tensor of shape (N, C, H, W)
    :param distance_metric: 'L2' or 'L1'
    :return: 1d tensor of shape (N, )
    """
    if distance_metric == 'L2':
        return torch.norm(torch.norm(targer_image - source_images, dim=(2, 3)), dim=1)
    elif distance_metric == 'L1':
        return torch.norm(torch.norm(targer_image - source_images, p=1, dim=(2, 3)), p=1, dim=1)
    else:
        raise ValueError("distance_metric must be 'L1' or 'L2'")
# %%

def vec_similarity(v1, v2, similarty_metric=None):
    if similarty_metric is None:
        raise ValueError("similarty_metric is should be a function that caculate the similarity between two vectors")
    if similarty_metric == "cosine" or similarty_metric == "cosine_req":
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    elif similarty_metric == "MSE":
        # caculate the MSE between two vectors
        return -np.mean((v1 - v2) ** 2)
    
 
#%% the asumptiopn is that element i of bost vector are coresponing to the same unit in the population. So we want find sub units that cause most the change in the cosine similarity
# we will use the falowing funtion to caculate the change in the cosine similarity for each unit ablation
def  unit_ablation_efect_on_similarity(v1, v2, unit_id, similarty_metric=None):
    if unit_id >= len(v1):
        raise ValueError("unit_id is out of range")
    if similarty_metric is None:
        raise ValueError("similarty_metric is should be a function that caculate the similarity between two vectors")
    v1_ablated = np.delete(v1, unit_id)
    v2_ablated = np.delete(v2, unit_id)
    return vec_similarity(v1, v2, similarty_metric) -\
            vec_similarity(v1_ablated, v2_ablated, similarty_metric)


# In the first hurstic we will rank all units with they efect on the cosine similarity and then we will select first 16 units that cause the most change in the cosine similarity

def select_sub_units(v1, v2, sub_population_size, similarty_metric=None):
    if sub_population_size > len(v1):
        raise ValueError("sub_population_size is bigger than the population size")
    if similarty_metric is None:
        raise ValueError("similarty_metric is should be a function that caculate the similarity between two vectors")
    unit_ids = np.arange(0, len(v1))
    # sort units by their effect on the similarity matric
    unit_ids = sorted(unit_ids, key=lambda unit_id: unit_ablation_efect_on_similarity(v1, v2, unit_id, similarty_metric), reverse=True)
    # lest return the best and the worst and random selection of units
    return unit_ids[:sub_population_size], unit_ids[-sub_population_size:], np.random.choice(unit_ids, sub_population_size, replace=False)

def get_a_file_name():
    timestamp = int(time.time())
    random_value = np.random.randint(1E7)
    return f"{timestamp}_{random_value}"

def print_tensor(imtsr, save_dir_root, meta_data_df, image_format = "jpg", PILimage=False, **kwargs):
    if (imtsr is not None) and (meta_data_df is not None):
        file_name = get_a_file_name()
        if not PILimage:
            imtsr = ToPILImage()(imtsr)
        imtsr.save(join(save_dir_root, f"{file_name}.{image_format}"))
        new_row = list()
        for column_name in meta_data_df.columns:
            if column_name not in kwargs:
                kwargs[column_name] = None
            new_row.append(kwargs[column_name])
        meta_data_df.loc[file_name] = new_row

def print_fig(fig, save_dir_root, meta_data_df, image_format="jpg", **kwargs):
    if (fig is not None) and (meta_data_df is not None):
        file_name = get_a_file_name()
        fig.savefig(join(save_dir_root, f"{file_name}.{image_format}"))
        new_row = list()
        for column_name in meta_data_df.columns:
            if column_name not in kwargs:
                kwargs[column_name] = None
            new_row.append(kwargs[column_name])
        meta_data_df.loc[file_name] = new_row
def data_saver(data_dic, save_dir_root, meta_data_df, **kwargs):
    if (data_dic is not None) and (meta_data_df is not None):
        file_name = get_a_file_name()
        np.savez(join(save_dir_root, f"{file_name}.npz"), **data_dic)
        new_row = list()
        for column_name in meta_data_df.columns:
            if column_name not in kwargs:
                kwargs[column_name] = None
            new_row.append(kwargs[column_name])
        meta_data_df.loc[file_name] = new_row

def fr_estimatir(scorer, G, unit_tsridx_dict, layer_name, input_size,show_fig=False, gan_name="fc6", singlecol=True):
    fitdict_dict = {}
    radius_list = []
    total_ampmap = None
    for layer in layer_name:
        unitslice = (unit_tsridx_dict[layer][0],
                    unit_tsridx_dict[layer][1][0],
                    unit_tsridx_dict[layer][2][0])
        gradAmpmap = GAN_grad_RF_estimate(G, scorer.model, layer, unitslice, input_size=input_size,
                                    device="cuda", show=show_fig, reps=100, batch=1, gan_type=gan_name)
        if singlecol:
            fitdict_dict[layer] = fit_2dgauss(gradAmpmap) #TODO: maby is not the best way to do it, maby you should add them
            radius_list.append((fitdict_dict[layer]["sigma_x"]**2) + (fitdict_dict[layer]["sigma_y"]**2))
        else:
            if total_ampmap is None:
                total_ampmap = gradAmpmap
            else:
                total_ampmap += gradAmpmap
            
    if singlecol:
        # fint reseptive fild with the biggest radius and return it coresponding dict
        return fitdict_dict[layer_name[np.argmax(radius_list)]]
    else:
        fitdict = dict()
        print(gradAmpmap.max())
        fitdict["fitmap"] = gradAmpmap/gradAmpmap.max()
        return fitdict
