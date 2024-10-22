import sys
import os
from argparse import ArgumentParser
import warnings

parser = ArgumentParser()
parser.add_argument("--net", type=str, default="alexnet", help="Network model to use for Image distance computation")
parser.add_argument("--gan_name", type=str, default="BigGAN", help="GAN model name")
parser.add_argument("--layers_short", type=str, default="conv5", help="shortcut for the layer name")
parser.add_argument("--popsize", type=int, default=256, help="Number of units in the population recording")
parser.add_argument("--sampling_strategy", type=str, default="random", choices=["random", "most"], help="select units randomly or based on their activation")

parser.add_argument("--score_method", type=str, default= "MSE",
    choices=["cosine", "MSE"], help="Objective function to assess the population response patteren similarity")

if sys.platform == "linux":
    save_data_main_root = r"/n/scratch3/users/a/ala2226/BigGAN_reconstruction_O2_preprocessed"
    data_root = r"/n/scratch3/users/a/ala2226/BigGAN_reconstruction_O2_080123"
    sys.path.append(r"/home/ala2226/Cosine-Project") 
    init_img_path = os.path.join(r"/n/scratch3/users/a/ala2226/BigGAN_reconstruction_O2_080123", r"init_img.jpg")
    RF_path = r"/n/scratch3/users/a/ala2226/insilico_experiments/rf_filters"
else:
    save_data_main_root = r"C:\Data\cosine\insilico_experiments\BigGAN_preprocesssing_test1"
    data_root = r"N:\PonceLab\Users\Alireza\insilico_experiments\Alexnet_remonstration_across_different_layer_062023\raw_data\cross_layer_recording_o2_062823"
    sys.path.append(r"C:\Users\Alireza\Documents\Git\Cosine-Project")
    init_img_path = os.path.join(r"N:\PonceLab\Users\Alireza\insilico_experiments\BigGAN_expriment_processed_data_and_results\init_image",  r"init_img.jpg")
    RF_path = r"N:\PonceLab\Users\Alireza\insilico_experiments\Alexnet_remonstration_across_different_layer_062023\post_processed\rf_filters"

os.makedirs(save_data_main_root, exist_ok=True)

if __name__=="__main__":
    import glob
    from PIL import Image
    from torchvision.transforms import ToTensor
    import numpy as np
    sys.path.append(r"C:\Users\Alireza\Documents\Git\Cosine-Project")
    from core.utils.func_lib import *
    from inSilico_experiments.utils.pothook_analysis_lib import *
    import pandas as pd
    
    # Set parameters
    args = parser.parse_args() 
    net_name = args.net
    layer_short = args.layers_short
    gan_name = args.gan_name
    popsize = args.popsize
    metric_name = args.score_method
    pd_key="expriment_meta_data_df"

    
    init_img = Image.open(init_img_path)
    init_img = ToTensor()(init_img)

    meta_data_file_root = os.path.join(data_root, "meta_data_files")
    # read .h5 files in the data_root folder with pattern 
    # example pattern = f"meta_data_alexnet_most_fc6_conv53_2_MSE_*.h5"
    pattern = f"meta_data_{net_name}_{args.sampling_strategy}_{gan_name}_{layer_short}_{popsize}_{metric_name}_*.h5"
    meta_data_file_path_list = glob.glob(os.path.join(meta_data_file_root, pattern))
    meta_data_df_batch = pd.DataFrame()
    for file_path in meta_data_file_path_list:
        metadata_df = pd.read_hdf(file_path, key=pd_key)

        file_name = file_path.split(os.sep)[-1]
        data_folder_name = file_name.split("_")[1]
        for si in file_name.split("_")[2:-1]:
            data_folder_name = data_folder_name +("_"+si)
        metadata_df["data_root"] = os.path.join(data_root, data_folder_name)
        print("number of repited index in the metadata data frame: ", metadata_df.index.duplicated().sum())
        print(metadata_df.shape)

        # add a new column to the dataframe to store the l2 distance between the generated image and the target image
        metadata_df["sim_index"] = np.nan
        metadata_df["l2_distance"] = np.nan
        row_count = 0
        # loop over all rows of the meta data dataframe
        print(f"processing {len(metadata_df[metadata_df['output_type'] == 'best_gen_imgs_RF_masked'])} rows")
        for i, row in metadata_df.iterrows():
            if row["output_type"] == "best_gen_imgs_RF_masked":
                # load the image as tensor
                img_path = glob.glob(os.path.join(row["data_root"], f"{i}.jpg"))[0]
                img = Image.open(img_path)
                gen_image_tensor = ToTensor()(img)
                # find target image id and load the image as tensor
                i_target = find_target_image_id(metadata_df, row)
                # load the target image as tensor
                img_path = glob.glob(os.path.join(row["data_root"], f"{i_target}.jpg"))[0]
                img = Image.open(img_path)
                target_image_tensor = ToTensor()(img)    

                if row["layer_short"] == "conv5432":
                    layer_short = "conv5"
                    pop_size = row["pop_size"]*4
                elif row["layer_short"] == "conv53":
                    layer_short = "conv5"
                    pop_size = row["pop_size"]*2
                else:
                    layer_short = row["layer_short"]
                    pop_size = row["pop_size"]
                
                RF_map = np.load(os.path.join(RF_path, f"{layer_short}_{pop_size}.npz"))

                RF_filter = RF_map["fitmap"] > RF_map["fitmap"][int(RF_map["xo"]+(1.5*RF_map["sigma_x"])), int(RF_map["yo"]+(1.5*RF_map["sigma_y"]))]
                init_img_rf_masked =\
                    (torch.from_numpy(np.absolute(RF_map["fitmap"][None,:,:])) / RF_map["fitmap"].max()) *\
                    init_img
                # calculate the l2 distance between the generated image and the target image and store it in the dataframe
                try:
                    metadata_df.loc[i, "sim_index_l1"] = \
                        sim_index_l1(target_image_tensor, gen_image_tensor, init_img_rf_masked, RFfilter=np.array(RF_filter))
                    metadata_df.loc[i, "sim_index_l2"] = \
                        sim_index_l2(target_image_tensor, gen_image_tensor, init_img_rf_masked, RFfilter=np.array(RF_filter))
                    metadata_df.loc[i, "l2_distance"] = \
                        l2_distance(target_image_tensor, gen_image_tensor).item()
                    metadata_df.loc[i, "l2_distance_normalized"] = \
                        normalized_l2(target_image_tensor, gen_image_tensor, RFfilter=np.array(RF_filter))
                    
                except:
                    print(f"error in calculating distances for raw: {i}")
                #get_similarty_score_by_CNN(gen_image_tensor, target_image_tensor, scorer_dict, metadata_df, i)
                row_count = row_count + 1
                if row_count % 100 == 0:
                    print(f"processed {row_count} rows...")
        print(f"all {row_count} rows have been processed!")
        os.makedirs(os.path.join(save_data_main_root, "metadata_dfs"), exist_ok=True)
        metadata_df.to_hdf(os.path.join(save_data_main_root, "metadata_dfs", f"preprocessed_{file_name}"), key=pd_key)
