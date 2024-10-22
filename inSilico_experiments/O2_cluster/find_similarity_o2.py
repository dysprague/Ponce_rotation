import sys
import os
import pandas as pd
from argparse import ArgumentParser
import glob
import shutil
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch
import numpy as np

if sys.platform == "linux":
    data_path = r"/n/scratch3/users/a/ala2226/image_similarity_062623"
    sys.path.append(r"/home/ala2226/Cosine-Project") #TODO: the path to the codebase you are running on the cluster
else:
    data_path = r"N:\PonceLab\Users\Alireza\insilico_experiments\Alexnet_remonstration_across_different_layer_062023\post_processed"
    sys.path.append(r"C:\Users\Alireza\Documents\Git\Cosine-Project")


from core.utils.func_lib import *
from core.utils.CNN_scorers import TorchScorer
from inSilico_experiments.utils.pothook_analysis_lib import *

metadata_df = pd.read_hdf(os.path.join(data_path, "metadata_random_df_o2.h5"), key="metadata_df")
# add a new column to the dataframe to store the l2 distance between the generated image and the target image
metadata_df["l2_distance"] = np.nan
row_count = 0
# loop over all rows of the meta data dataframe
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
        # plot the generated image and the target image in a figure to make sure that they are the same
        '''fig, ax = plt.subplots(1, 2)
        ax[0].imshow(ToPILImage()(gen_image_tensor))
        ax[1].imshow(ToPILImage()(target_image_tensor))
        plt.show()'''
        # calculate the l2 distance between the generated image and the target image and store it in the dataframe
        try:
            metadata_df.loc[i, "l2_distance"] = l2_distance(gen_image_tensor, target_image_tensor).item()
        except:
            print(f"error in calculating l2 distance for raw: {i}")
        #get_similarty_score_by_CNN(gen_image_tensor, target_image_tensor, scorer_dict, metadata_df, i)
        row_count = row_count + 1
        
metadata_df.to_hdf(os.path.join(data_path, "metadata_random_df_with_similarity_o2.h5"), key="metadata_df")