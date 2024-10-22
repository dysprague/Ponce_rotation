from datetime import datetime
import os
import pandas as pd
import sys
sys.path.append(r"C:\Users\Alireza\Documents\Git\Cosine-Project")
from core.utils.func_lib import *
if __name__=="__main__":
    #%%
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M")
    from core.utils.GAN_utils import upconvGAN
    from core.utils.Optimizers import CholeskyCMAES
    from core.utils.grad_RF_estim import fit_2dgauss, GAN_grad_RF_estimate
    from core.utils.CNN_scorers import TorchScorer, resize_and_pad_tsr
    #%% Set parameters
    refimgdir = r"C:\Data\cosine\insilico_experiments\papulation_size_effect\data\data_sample"
    exproot = r"C:\Data\cosine\insilico_experiments\papulation_size_effect\results\result_real_%s" % date_time_str
    os.makedirs(exproot, exist_ok=True)
    score_methodlist = ["Chebyshev", "L1", "MSE", "cosine", "Correlation", "dot", "Spearman"]
    population_size = [100, 200]#[1, 25, 50, 100, 150, 200, 250]
    GAN_name = "fc6"
    net_name = "alexnet"
    layer_name = ".features.Conv2d10"
    layer_short = "conv5"
    #%% Load the reference images
    refimgnms, refimgtsr = load_ref_imgs(
        imgdir=refimgdir, preprocess_type='center_crop', image_size=227)
    #%% Set up the scorer and the population
    scorer = TorchScorer(net_name)
    module_names, module_types, module_spec = get_module_names(scorer.model, (3, 227, 227), "cuda", False)

    # %% set up the GAN
    G = upconvGAN(GAN_name).cuda()
    G.requires_grad_(False)
    code_length = G.codelen
    # %% pandas dataframe to save the results of the experiment with filed for each score method and population
    # size and images used
    target_gen_img_distance_df = pd.DataFrame(columns=["target_img", "score_method", "population_size",
                                                       "target_gen_img_distance"])
    objective_score_df = pd.DataFrame(columns=["target_img", "score_method", "population_size", "objective_score"])
    #%% Run the experiment
    for popsize in population_size:
        #%% make a directory for the population size
        expdir_pop = os.path.join(exproot, f"popsize-{popsize}")
        os.makedirs(expdir_pop, exist_ok=True)
        #%% Set the population
        scorer = TorchScorer(net_name)
        unit_mask_dict, unit_tsridx_dict = set_random_population_recording(scorer, [layer_name], popsize=popsize)
        # Encode a population of images to set the normalizer and mask.
        ref_actmat, _ = encode_image(scorer, refimgtsr, key=layer_name,
                                  RFresize=True, corner=(20, 20), imgsize=(187, 187))
        popul_m, popul_s = set_normalizer(ref_actmat)
        popul_mask = set_popul_mask(ref_actmat)

        # %% calculate the covariance matrix of the reference population
        ref_actmat_ZS = (ref_actmat - popul_m) / popul_s
        ref_act_mat_cov = np.cov(ref_actmat_ZS.T)

        #%% receptive field estimation
        unitslice = (unit_tsridx_dict[layer_name][0],
                     unit_tsridx_dict[layer_name][1][0],
                     unit_tsridx_dict[layer_name][2][0])
        gradAmpmap = GAN_grad_RF_estimate(G, scorer.model, layer_name, unitslice, input_size=(3, 227, 227),
                                      device="cuda", show=True, reps=100, batch=1)
        fitdict = fit_2dgauss(gradAmpmap, f"{net_name}-{popsize}-" + layer_short, outdir=None, plot=False)

        #%% Run the evolution
        for imgid in range(len(refimgnms)):
            # Select target image and add target vector.
            targnm, target_imgtsr = refimgnms[imgid], refimgtsr[imgid:imgid + 1]
            targ_actmat, target_imgtsr_resized = encode_image(scorer, target_imgtsr, key=layer_name,
                                       RFresize=True, corner=(20, 20), imgsize=(187, 187))
            targlabel = os.path.splitext(targnm)[0]
            # organize data with the targetlabel
            expdir_img = os.path.join(expdir_pop, "rec_%s"%targlabel)
            os.makedirs(expdir_img, exist_ok=True)
            ToPILImage()(target_imgtsr_resized[0]).save(join(expdir_img, "targetimg.png"))
            #%% RF masking of the target image
            target_imgtsr_resized_RF_masked =\
                (torch.from_numpy(np.absolute(fitdict["fitmap"][None,:,:])) / fitdict["fitmap"].max()) *\
                target_imgtsr_resized
            ToPILImage()(target_imgtsr_resized_RF_masked[0]).save(join(expdir_img, "targetimg_RF_masked.png"))
            for score_method in ["euclidean_req"]:
                expdir_meth = os.path.join(expdir_img, "method_%s" % score_method)
                os.makedirs(expdir_meth, exist_ok=True)
                explabel = "%s-%s-%s-popsize%d"%(targlabel, score_method, GAN_name, popsize)
                objfunc = set_objective(score_method, targ_actmat, popul_mask=None, popul_m=popul_m,
                                        popul_s=popul_s, VI=ref_act_mat_cov)
                print("Running %s" % explabel)
                optimizer = CholeskyCMAES(code_length, population_size=None, init_sigma=3,
                                init_code=np.zeros([1, code_length]), Aupdate_freq=10,
                                maximize=True, random_seed=None, optim_params={})
                codes_all, scores_all, actmat_all, generations, RND, last_gem_img, last_gem_img_maske =\
                    run_evol(scorer, objfunc, optimizer, G, reckey=layer_name, label=explabel, savedir=expdir_meth,
                    steps=100, RFresize=True, corner=(20, 20), imgsize=(187, 187), RF_mask=fitdict["fitmap"])
                figh = visualize_popul_act_evol(actmat_all, generations, targ_actmat)
                figh.savefig(join(expdir_meth, "popul_act_evol_%s_%d.png" % (explabel, RND)))
                targe_gen_img_distance = get_image_distance_tensor(
                    target_imgtsr_resized_RF_masked[0], last_gem_img_maske)
                # save the targe_gen_img_distance mean in the dataframe for each score method and population size AND image
                # used

                new_row = {"target_img": targlabel,
                           "score_method": score_method,
                           "population_size": popsize,
                           "target_gen_img_distance": targe_gen_img_distance.mean().item()}

                target_gen_img_distance_df = pd.concat(
                    [target_gen_img_distance_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

                # save the objective score in the dataframe for each score method and population size AND image
                # used
                new_row = {"target_img": targlabel,
                            "score_method": score_method,
                            "population_size": popsize,
                            "objective_score": scores_all[-1]}
                objective_score_df = pd.concat(
                    [objective_score_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)

    target_gen_img_distance_df.to_csv(
          os.path.join(exproot, f"target_gen_img_distance_df.cvs"), index=False)
    objective_score_df.to_csv(
        os.path.join(exproot, f"objective_score_df.cvs"), index=False)

