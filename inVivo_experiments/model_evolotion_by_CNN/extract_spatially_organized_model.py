
from argparse import ArgumentParser
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import mat73
from torchvision.transforms import ToPILImage, PILToTensor, ToTensor
import matplotlib.image as mpimg
from PIL import Image
from scipy import stats, ndimage
from torchvision.utils import make_grid
from core.utils.CNN_scorers import TorchScorer
from core.utils.func_lib import *
from core.utils.basic_functions import *
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.dummy import DummyRegressor
from sklearn.metrics import make_scorer, mean_squared_error

parser = ArgumentParser()
parser.add_argument("--animal_name", type=str, default="Caos", help="What is the name of the animal")
parser.add_argument("--net_name", type=str, default="alexnet", help="What is the name of the CNN model to model the responses")


#parser.add_argument("--max_iter", type=int, default=int(5E5), help="Number gradient descent iterations")
#parser.add_argument("--batch_num", type=int, default=1, help="batch number to process")

if sys.platform == "linux":
    save_data_main_root = r"/n/scratch/users/a/ala2226/gan_inversion_results_fc6  
    data_root = r"/n/scratch/users/a/ala2226/...
    sys.path.append(r"/home/ala2226/Cosine-Project") 
else:
    save_data_main_root = r"C:\Users\Alireza\OneDrive - Harvard University\Documents\coisne_results\feature_mask_estimation"
    data_root = r"C:\Users\Alireza\OneDrive - Harvard University\Documents\cosine_preprocess_data"
    server_init = r"N:\PonceLab"
    sys.path.append(r"C:\Users\Alireza\Documents\Git\Cosine-Project")

os.makedirs(save_data_main_root, exist_ok=True)

if __name__=="__main__":
    # let get log of the time
    t_save = time.time()
    from core.utils.image_analysis import *
    from core.utils.basic_functions import *

    animal_name = "Caos"
    net_name = 'alexnet'



    save_root = os.path.join(save_data_main_root, 'feature_mask_estimation', animal_name)
    os.makedirs(save_root, exist_ok=True)

    # Define the paths to the data
    proto_exp_path = os.path.join(data_root, "evolution", f"unit_proto_{animal_name}")
    # List .mat files in each directory
    proto_exp_files = list_mat_files(proto_exp_path)

    
    layer_list = ['.features.Conv2d3']

    # let's do the cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    #reg = linear_model.LinearRegression()
    lasso = linear_model.Lasso()
    baseline_model = DummyRegressor(strategy='mean')


    proto_exp_i = 21 #TODO: change this to a loop

    proto_exp_data = load_mat_data(os.path.join(proto_exp_path, proto_exp_files[proto_exp_i]))
    proto_exp_info = extract_info_proto_exp(proto_exp_files[proto_exp_i])
    exp_name_common = '-'.join(proto_exp_info['exp_name'].split('-')[0:2])

    chan_id = int(proto_exp_data['evoled_chan'])
    unit_id = int(proto_exp_data['evoled_unit'])

    # the scram exp data file name is formated like this [exp_name]_exp_idXXX_chanXX_unitX.mat for example like Caos-30112023-008_expId022_chan71_unit2
    # let's load the coresponded scram exp data which should be [exp_name_common]-*_chan[chan_id]_unit[unit_id].mat
    scram_exp_file = [f for f in scram_exp_files if (f.count(f"chan{chan_id:02d}") and f.count(f"unit{unit_id}") and f.count(exp_name_common))]
    # continue if the file is not found
    if (len(scram_exp_file) == 0) or (proto_exp_data['p_evol'] > .01):
        print(f"Scrambled file not found for {proto_exp_files[proto_exp_i]}")

    scram_exp_file = scram_exp_file[0]
    scram_exp_info = mat73.loadmat(os.path.join(scram_exp_path, scram_exp_file))
    proto_size = scram_exp_info['img_size']
    proto_pos = scram_exp_info['img_pos']
    protoRespMean = scram_exp_info['protoRespMean']
    exp_date = scram_exp_file.split('_')[0].split('-')[1]

    proto_img_top_mean = (proto_exp_data['top_mean_img_selected']) # TODO: if you add somthing here you should resize it
    proto_img_scram = (scram_exp_info['final_gen_img'])
    init_img = (proto_exp_data['init_img'])
    proto_mask_exp = scram_exp_info['overlapped_mask_exp']
    proto_mask_lin = scram_exp_info['overlapped_mask_lin']
    proto_mask_lin_smoothed = ndimage.gaussian_filter(proto_mask_lin, 25)
    # if mask is not the size of image_size then resize it
    iamge_size = 256
    if proto_mask_exp.shape[0] != iamge_size:
        proto_mask_exp = cv2.resize(proto_mask_exp, (iamge_size, iamge_size))
    if proto_mask_lin.shape[0] != iamge_size:
        proto_mask_lin = cv2.resize(proto_mask_lin, (iamge_size, iamge_size))
    if proto_mask_lin_smoothed.shape[0] != iamge_size:
        proto_mask_lin_smoothed = cv2.resize(proto_mask_lin_smoothed, (iamge_size, iamge_size))
    # if the proto_img is not the size of image_size then resize it
    if proto_img_top_mean.shape[0] != iamge_size:
        proto_img_top_mean = cv2.resize(proto_img_top_mean, (iamge_size, iamge_size))
    if proto_img_scram.shape[0] != iamge_size:
        proto_img_scram = cv2.resize(proto_img_scram, (iamge_size, iamge_size))

    proto_img_top_mean = uint8_image_to_pytorch(proto_img_top_mean)
    proto_img_scram = uint8_image_to_pytorch(proto_img_scram)
    init_img = uint8_image_to_pytorch(init_img)


    # let get all imaes 
    stimuli_path = proto_exp_data['stimuli_path']
    images_tensor, image_name_lisst = load_all_images(stimuli_path)


    scorer = TorchScorer(net_name)
    unit_mask_dict, unit_tsridx_dict = set_all_unit_population_recording(scorer, layer_list)
    encoded_first_image_batch, _ = encode_image(scorer, images_tensor, key=layer_list,
                                                RFresize=False, cat_layes=False)
    encoded_first_image_batch = np.array(encoded_first_image_batch)
    scorer.cleanup()


    layer_list_idx = 0
    chan_max = unit_tsridx_dict[layer_list[layer_list_idx]][0].max()
    H_max = unit_tsridx_dict[layer_list[layer_list_idx]][1].max()
    W_max = unit_tsridx_dict[layer_list[layer_list_idx]][2].max()


    # let find the coresponing image encoding to each resonce, we should use all_img_name
    # to find the index of the image in the images_tensor
    mised_img = 0
    cell_resp = proto_exp_data['evoke_resp'][(proto_exp_data['spikeID'] == proto_exp_data['evoled_chan']) & (proto_exp_data['unitID'] == proto_exp_data['evoled_unit']), :].squeeze()
    cell_resp_coresponing_img_encoded = np.nan * np.ones((cell_resp.shape[0], np.shape(encoded_first_image_batch)[2]))

    for ii, img in enumerate(proto_exp_data['all_img_name']):
        # check if wher the img is contained in the image_name_list, img is a string and image_name_list is numpy array of strings
        # let's find the index of the img name is contained in the image_name_list
        idx = [i for i, name in enumerate(image_name_lisst) if name in img]
        if len(idx) > 0:
            idx = idx[0]
            cell_resp_coresponing_img_encoded[ii, :] = encoded_first_image_batch[layer_list_idx, idx, :]
        else:
            mised_img += 1
    # let's remove the nan values fron the cell_resp_coresponing_img_encoded and cell_resp
    idx = ~np.isnan(cell_resp_coresponing_img_encoded).any(axis=1)
    cell_resp_coresponing_img_encoded = cell_resp_coresponing_img_encoded[idx, :]
    cell_resp = cell_resp[idx] # TO DO -> BAD IDEA TO REMOVE THE MISSED IMAGES IN THE SAME VARIABLE
    print(f"Missed images: {cell_resp_coresponing_img_encoded.shape[0] - np.sum(idx)}")


    layer_list_idx = 0
    chan_max = unit_tsridx_dict[layer_list[layer_list_idx]][0].max()
    H_max = unit_tsridx_dict[layer_list[layer_list_idx]][1].max()
    W_max = unit_tsridx_dict[layer_list[layer_list_idx]][2].max()

    r2_mask_grid_mean = np.zeros([H_max+1, W_max+1])
    r2_mask_grid_folds = np.zeros([H_max+1, W_max+1, kf.get_n_splits()])
    r2_mask_permuted_grid_mean = np.zeros([H_max+1, W_max+1])
    r2_mask_permuted_grid_folds = np.zeros([H_max+1, W_max+1, kf.get_n_splits()])

    for ih in range(H_max+1):
        for iw in range(W_max+1):
            idx_mask = np.where((unit_tsridx_dict[layer_list[layer_list_idx]][1] == ih) & (unit_tsridx_dict[layer_list[layer_list_idx]][2] == iw))[0]
            cell_resp_coresponing_img_encoded_mask = cell_resp_coresponing_img_encoded[:, idx_mask]       
            scores_real = cross_val_score(lasso, cell_resp_coresponing_img_encoded_mask, cell_resp, cv=kf)
            # let permute the cell_resp and repeat the cross_val_score
            cell_resp_permuted = np.random.permutation(cell_resp)
            scores_permuted = cross_val_score(lasso, cell_resp_coresponing_img_encoded_mask, cell_resp_permuted, cv=kf)
            
            r2_mask_grid_mean[ih, iw] = np.mean(scores_real)
            r2_mask_grid_folds[ih, iw, :] = scores_real
            r2_mask_permuted_grid_mean[ih, iw] = np.mean(scores_permuted) #TODO: we need 10 of this :((
            r2_mask_permuted_grid_folds[ih, iw, :] = scores_permuted

