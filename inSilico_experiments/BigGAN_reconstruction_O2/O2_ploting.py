#%% importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import pickle
sys.path.append(r"C:\Users\Alireza\Documents\Git\Cosine-Project")
from inSilico_experiments.utils.pothook_analysis_lib import *
from torchvision.transforms import ToTensor, ToPILImage, Pad, Compose
from torchvision.utils import make_grid
from core.utils.CNN_scorers import resize_and_pad_tsr

#%% load the metadata dataframe
pd_key="expriment_meta_data_df"
data_root = r"/n/scratch3/users/a/ala2226/BigGAN_reconstruction_O2_preprocessed"
meta_data_df = pd.DataFrame()
for file in os.listdir(data_root):
    if file.endswith(".h5"):
        meta_data_df = pd.concat([meta_data_df, pd.read_hdf(os.path.join(data_root, file), key=pd_key)], axis=0)

#%% print one of the target images and the its recontruction for each layer and population size (load all regen images)
image_name_list = metadata_df_cosine["trget_imge_name"].unique()
layer_short_list = ["conv5", "conv4", "conv3", "conv2"]
similarity_metric_list = ["cosine", "MSE"]
pop_size_list = [2, 8, 32, 128]
RF_treshold = 2
pop_resampling_id = 0
smpling_type = "random"

output_type = "target_img_RF_masked"
gen_image_type = "best_gen_imgs_RF_masked"

# load the recontruction image for each layer and plot them in a grid 
# each row is a layer and each column is a population size
save_path = os.path.join(save_root, f"recontruction_images_{smpling_type}")
for similarity_metric in similarity_metric_list:
    save_path = os.path.join(save_path, similarity_metric)
    os.makedirs(save_path, exist_ok=True)
    for image_name in image_name_list:
        fig, ax = plt.subplots(len(layer_short_list), len(pop_size_list)+1, figsize=(20, 20))
        for i, layer_short in enumerate(layer_short_list):
            target_image = load_image_tsr(metadata_df_cosine, trget_imge_name=image_name,
                                        layer_short=layer_short, output_type=output_type, pop_size=pop_size_list[-2],
                                        sub_pop_type="most", pop_resampling_id=pop_resampling_id)
                                        
            rf_filter = np.load(os.path.join(data_path, "rf_filters", f"{layer_short}_{pop_size_list[-2]}.npz"))
            target_image_croped = image_rf_crop(target_image, (int(rf_filter["sigma_y"]*RF_treshold*2),
                                                            int(rf_filter["sigma_x"]*RF_treshold*2)))
            # show the target image
            ax[i, 0].imshow(ToPILImage()(target_image_croped))
            ax[i, 0].set_title(f"target image", fontsize=23, fontweight="bold")
            # set x label
            ax[i, 0].set_ylabel(f"{layer_short}", fontsize=27, fontweight="bold")
            # remove just y axix
            ax[i, 0].set_xticks([])
            ax[i, 0].set_yticks([])
            for j, pop_size in enumerate(pop_size_list):
                try:
                    # load the recontruction image for each gen_rerun_id and append it to a tensor
                    rcon_score_list = []
                    for gen_rerun_id in range(5):
                        recontruction_image = load_image_tsr(metadata_df_cosine, trget_imge_name=image_name, layer_short=layer_short,
                                                            output_type=gen_image_type, pop_size=pop_size, gen_rerun_id=gen_rerun_id,
                                                            similarity_metric=similarity_metric, sub_pop_type=smpling_type,
                                                            pop_resampling_id=pop_resampling_id)

                        # load the RF mask
                        rf_filter = np.load(os.path.join(data_path, "rf_filters", f"{layer_short}_{pop_size}.npz"))
                        # crop the recontruction image
                        recontruction_image_croped = image_rf_crop(recontruction_image, (int(rf_filter["sigma_y"]*RF_treshold*2),
                                                                                        int(rf_filter["sigma_x"]*RF_treshold*2)))
                        rcon_score_list.append(column_inquary("score", metadata_df_cosine, trget_imge_name=image_name, layer_short=layer_short,
                                                        output_type=gen_image_type, pop_size=pop_size, gen_rerun_id=gen_rerun_id,
                                                        similarity_metric=similarity_metric, sub_pop_type=smpling_type,
                                                        pop_resampling_id=pop_resampling_id))
                        if gen_rerun_id == 0:
                            recontruction_image_tensor = recontruction_image_croped.unsqueeze(0)
                        else:
                            recontruction_image_tensor = torch.cat((recontruction_image_tensor, recontruction_image_croped.unsqueeze(0)), dim=0)
                    rcon_score = np.mean(rcon_score_list)        
                    recontruction_image_grid = make_grid(recontruction_image_tensor, padding= 2, pad_value= 1, nrow=2)
                    # show the recontruction image           
                    ax[i, j+1].imshow(ToPILImage()(recontruction_image_grid))
                    # title with score and pop_size and bold
                    ax[i, j+1].set_title(f"score: {rcon_score:.2f} ", fontsize=23, fontweight="bold")
                except:
                    pass
                ax[i, j+1].axis("off")
        plt.tight_layout()
        # save the figure  
        plt.savefig(os.path.join(save_path, f"{image_name}_{similarity_metric}_recontruction_all_gen_{smpling_type}.png"), bbox_inches="tight", pad_inches=0)
        plt.close()

#%% assement of flip invariance
# preaper the layout for the incvariance plot
import matplotlib as mpl
color_map_name_cosine = "jet"
pading_cmap_cosine=mpl.colormaps[color_map_name_cosine]
color_map_name_MSE = "jet"
pading_cmap_MSE=mpl.colormaps[color_map_name_MSE]

def addhock_normalization(inval, minval, maxval):
    return (inval - minval) / (maxval - minval)

cosine_min = .75
cosine_max = 1
MSE_min = -25
MSE_max = 3
image_name_list = metadata_df_cosine["trget_imge_name"].unique()
layer_short_list = ["conv5", "conv4", "conv3", "conv2"]
similarity_metric_list = ["cosine", "MSE"]
smpling_type = "most"
pop_size = 128
RF_treshold = 2
output_type = "target_img_RF_masked"
gen_image_type = "best_gen_imgs_RF_masked"
save_path_root = os.path.join(save_root, "rotation_invariance")
save_path = save_path_root
os.makedirs(save_path, exist_ok=True)
padding = 3
for image_name in image_name_list:
    # make a sub plot grid of size (3, len(layer_short_list)) the ratio of the row is 3:1:3
    fig, ax = plt.subplots(3, len(layer_short_list), figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1, 3]})
    for i, layer_short in enumerate(layer_short_list):
        target_image = load_image_tsr(metadata_df_cosine, trget_imge_name=image_name,
                                    layer_short=layer_short, output_type=output_type, pop_size=pop_size, sub_pop_type=smpling_type)
        rf_filter = np.load(os.path.join(data_path, "rf_filters", f"{layer_short}_{pop_size}.npz"))
        target_image_croped = image_rf_crop(target_image, (int(rf_filter["sigma_y"]*RF_treshold*2),
                                                        int(rf_filter["sigma_x"]*RF_treshold*2)))
        # show the target image
        ax[1, i].imshow(ToPILImage()(target_image_croped))
        #ax[1, i].set_ylabel(f"target image", fontsize=23, fontweight="bold")
        ax[1, i].set_ylabel(f"{layer_short}", fontsize=23, fontweight="bold")
        # remove just y axix
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        # load the recontruction image for each gen_rerun_id and append it to a tensor
        j = 0
        for similarity_metric in similarity_metric_list:
            rcon_score_list = []
            for gen_rerun_id in range(5):
                recontruction_image = load_image_tsr(metadata_df_cosine, trget_imge_name=image_name, layer_short=layer_short,
                                                    output_type=gen_image_type, pop_size=pop_size, gen_rerun_id=gen_rerun_id,
                                                    similarity_metric=similarity_metric, sub_pop_type=smpling_type)
                # load the RF mask
                rf_filter = np.load(os.path.join(data_path, "rf_filters", f"{layer_short}_{pop_size}.npz"))
                # crop the recontruction image
                recontruction_image_croped = image_rf_crop(recontruction_image, (int(rf_filter["sigma_y"]*RF_treshold*2),
                                                                                int(rf_filter["sigma_x"]*RF_treshold*2)))
                score = column_inquary("score", metadata_df_cosine, trget_imge_name=image_name, layer_short=layer_short,
                                                output_type=gen_image_type, pop_size=pop_size, gen_rerun_id=gen_rerun_id,
                                                similarity_metric=similarity_metric, sub_pop_type=smpling_type)
                rcon_score_list.append(score)
                if similarity_metric == "cosine":
                    norm_score = addhock_normalization(score, cosine_min, cosine_max)
                    recontruction_image_croped = add_padding(recontruction_image_croped, int(padding+(padding/(i+1))), pading_cmap_cosine(norm_score)[:3])
                elif similarity_metric == "MSE":
                    norm_score = addhock_normalization(score, MSE_min, MSE_max)
                    recontruction_image_croped = add_padding(recontruction_image_croped, int(padding+(padding/(i+1))), pading_cmap_MSE(norm_score)[:3])
                if gen_rerun_id == 0:
                    recontruction_image_tensor = recontruction_image_croped.unsqueeze(0)
                else:
                    recontruction_image_tensor = torch.cat((recontruction_image_tensor, recontruction_image_croped.unsqueeze(0)), dim=0)
            rcon_score = np.mean(rcon_score_list)        
            recontruction_image_grid = make_grid(recontruction_image_tensor, padding= 3, pad_value= 1, nrow=2)
            # show the recontruction image           
            ax[j, i].imshow(ToPILImage()(recontruction_image_grid))
            ax[j, i].axis("off")
            j += 2
            # set tight layout
            
    plt.tight_layout()
    # save the figure 
    plt.savefig(os.path.join(save_path, f"{image_name}_invariance.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

