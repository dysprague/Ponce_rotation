import sys
import os
import pandas as pd
from argparse import ArgumentParser
import warnings

parser = ArgumentParser()
parser.add_argument("--net", type=str, default="alexnet", help="Network model to use for Image distance computation")
parser.add_argument("--layers", type=str, default=".features.Conv2d10", nargs="+", 
                     help="Network model to use for Image distance computation")
parser.add_argument("--layers_short", type=str, default="conv5", help="shortcut for the layer name")
parser.add_argument("--input_size", type=tuple, default=(3, 227, 227), help="net input image size")
parser.add_argument("--img_size", type=tuple, default=(147, 147), help="image size after padding")
parser.add_argument("--pading_size", type=tuple, default=(40, 40), help="padding size")


if sys.platform == "linux":
    exp_result_main_root = r"/n/scratch3/users/a/ala2226/BigGAN_reconstruction_O2_080123"
    refimgdir = r"/home/ala2226/val-imagenet"
    sys.path.append(r"/home/ala2226/Cosine-Project") #TODO: the path to the codebase you are running on the cluster

else:
    exp_result_main_root = r"N:\PonceLab\Users\Alireza\insilico_experiments\quest_of_image_manifould"
    refimgdir = r"N:\PonceLab\Stimuli\imagenet\val"
    sys.path.append(r"C:\Users\Alireza\Documents\Git\Cosine-Project")

os.makedirs(exp_result_main_root, exist_ok=True)

def count_files_with_formats(root_dir):
    formats = [".jpg", ".gif", ".png", ".tga", ".jpeg"]
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.lower().endswith(format) for format in formats):
                count += 1
    return count


if __name__=="__main__":
    #%%
    #import datetime
    from core.utils.GAN_utils import loadBigGAN, BigGAN_wrapper
    from core.utils.Optimizers import CholeskyCMAES
    from core.utils.CNN_scorers import TorchScorer
    from core.utils.func_lib import *
    from pytorch_pretrained_biggan import truncated_noise_sample
    import datetime
    #%% 
    # Set parameters
    args = parser.parse_args() 

    net_name = args.net
    layer_name = args.layers
    layer_short = args.layers_short
    input_size = args.input_size
    img_size = args.img_size
    pading_size = args.pading_size

    if not isinstance(layer_name, list):
        raise ValueError("layer_name must be a list of strings")
    
    #%% make a dictionary of the parameters
    trial_param_dict = {"net_name": net_name,
                    "layer_name": layer_name,
                    "layer_short": layer_short,
                    "input_size": input_size,
                    "pading_size": pading_size}
    #%% set up a pandas dataframe to save the inforamtions about each run
    expriment_meta_data_df = pd.DataFrame(columns=[ "img_name", "img_folder", "layer_name", "layer_short", "net_name", "img_size", "pading_size", "input_size"])
    #%% make name for expriment meta data frame file 
   
    # generate 5 digit random number for the meta data file name
    random_number = np.random.randint(10000, 99999)
    
    exp_file_name = f"data_{net_name}_{layer_short}"
    meta_file_name = f"meta_data_{net_name}_{layer_short}_{random_number}.h5"
    exp_result_root = os.path.join(exp_result_main_root, exp_file_name)
    # make a directory for the expriment
    os.makedirs(exp_result_root, exist_ok=True)
    # make the meta data directory
    os.makedirs(os.path.join(exp_result_main_root, "meta_data_files"), exist_ok=True)
    # get number of images in the refimgdir
    image_count = count_files_with_formats(refimgdir)
    print(f"number of images in the refimgdir: {image_count}")
    # get name of all folders in the refimgdir, just the folder not files
    img_folders = os.listdir(refimgdir)
    # Filter out only the folders
    img_folders = [item for item in img_folders if os.path.isdir(os.path.join(refimgdir, item))]
    num_of_all_image = len(img_folders)
    # set scorer ans population hock
    scorer = TorchScorer(net_name)
    unit_mask_dict, unit_tsridx_dict = set_all_center_unit_population_recording(scorer, layer_name)
    pop_size = 0
    for layer in layer_name:
        pop_size += len(unit_mask_dict[layer])
    # define the population activity tensor
    population_act_tensor = np.empty((image_count, pop_size))
    c = 0
    # Load the reference images
    

    for folder_name in img_folders:
        t1 = datetime.datetime.now()
        # get the path to the folder
        folder_path = os.path.join(refimgdir, folder_name)
        # lad the images in the folder
        refimgnms, refimgtsr = load_ref_imgs(
            imgdir=folder_path, preprocess_type='center_crop', image_size=227)
        targ_actmat, _ = encode_image(scorer, refimgtsr, key=layer_name,
                                    RFresize=True, corner=pading_size, imgsize=img_size)
        population_act_tensor[c : c+targ_actmat.shape[0], :] = targ_actmat
        c = c+targ_actmat.shape[0]
        propagate_data = {'img_name': refimgnms,
                        'img_folder': [folder_name] * targ_actmat.shape[0],
                        'layer_name': [layer_name] * targ_actmat.shape[0],
                        'layer_short': [layer_short] * targ_actmat.shape[0],
                        'net_name': [net_name] * targ_actmat.shape[0],
                        'img_size': [img_size] * targ_actmat.shape[0],
                        'pading_size': [pading_size] * targ_actmat.shape[0],
                        'input_size': [input_size] * targ_actmat.shape[0]}
        df_propagate = pd.DataFrame(propagate_data)
        expriment_meta_data_df = pd.concat([expriment_meta_data_df, df_propagate], ignore_index=True)
        # print the shape of the target activation matrix
    # save the population activity tensor
    save_path = os.path.join(exp_result_root, "encoded_centeral_col")
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "population_act_tensor.npy"), population_act_tensor)
    expriment_meta_data_df.to_hdf(os.path.join(save_path, "expriment_meta_data_df.h5"), key="expriment_meta_data_df", mode="w")
    print("Done loading the reference images")

# %%
