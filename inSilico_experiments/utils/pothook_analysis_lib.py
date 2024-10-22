import numpy as np
import scipy.stats as stats
import pandas as pd
import os
import glob
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize, CenterCrop
import torch
import shutil


def perform_linear_regression_with_sampling(X, y, sample_size = None):
    # Perform random sampling
    slope_list = list()
    intercept_list = list()
    r_value_list = list()
    p_value_list = list()
    std_err_list = list()
    
    for i in range(1000):
        if sample_size is None:
            sample_size = int(len(X)*0.8)
        random_indices = np.random.choice(len(X), sample_size, replace=False)
        X_sampled = X[random_indices]
        y_sampled = y[random_indices]

        # Calculate additional statistics using linregress from scipy.stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(X_sampled, y_sampled)

        slope_list.append(slope)
        intercept_list.append(intercept)
        r_value_list.append(r_value)
        p_value_list.append(p_value)
        std_err_list.append(std_err)
    
    return slope_list, intercept_list, r_value_list, p_value_list, std_err_list

def find_target_image_id(data_table, row, targettype="target_img_RF_masked", extrcollist=None):
    # find the target row in the data_table dataframe that have similar similar:
    # trget_imge_name	similarity_metric	pop_size	pop_resampling_id	gan_name	layer_short	net_name but it has output_type as "target_img_RF_masked"
    col_filter = (data_table["trget_imge_name"] == row["trget_imge_name"]) & (
        data_table["pop_size"] == row["pop_size"]) & (
        data_table["pop_resampling_id"] == row["pop_resampling_id"]) & (
        data_table["gan_name"] == row["gan_name"]) & (
        data_table["layer_short"] == row["layer_short"]) & (
        data_table["net_name"] == row["net_name"]) & (
        data_table["output_type"] == targettype) & (
        data_table["sub_pop_type"] == row["sub_pop_type"]) & (
        data_table["data_root"] == row["data_root"])
    if  extrcollist is not None:
        for col in extrcollist:
            col_filter = col_filter & (data_table[col] == row[col])
    target_row = data_table.loc[col_filter]
    if len(target_row) != 1:
        print(target_row)
        raise ValueError("The target row is not unique")
    return target_row.index[0]

def load_image_tsr(data_table, data_root=None, **kwargs):
     # find the row in the dataframe that match the input kwargs, the colums which are not mentioned in the kwargs are None
    row = data_table.loc[(data_table[list(kwargs)] == pd.Series(kwargs)).all(axis=1)]
    # if the row is empty return None
    if len(row) != 1:
        print(row)
        raise ValueError(f"the input kwargs: {kwargs} does not match any row in the dataframe or match more than one row")
    # if the row is not empty load the image
    else:
        if data_root is None:
            try:
                data_root = row["data_root"].values[0]
            except:
                raise ValueError("data_root is not in the dataframe and is not provided as an input")
            
        img_path = glob.glob(os.path.join(data_root, f"{row.index[0]}.jpg"))[0]
        img = Image.open(img_path)
        image_tensor = ToTensor()(img)
        return image_tensor
    
def coppy_image(data_table, distination_path, new_file_name=None, data_root=None, **kwargs):
     # find the row in the dataframe that match the input kwargs, the colums which are not mentioned in the kwargs are None
    row = data_table.loc[(data_table[list(kwargs)] == pd.Series(kwargs)).all(axis=1)]
    # if the row is empty return None
    if len(row) != 1:
        print(row)
        raise ValueError(f"the input kwargs: {kwargs} does not match any row in the dataframe or match more than one row")
    # if the row is not empty load the image
    else:
        if data_root is None:
            try:
                data_root = row["data_root"].values[0]
            except:
                raise ValueError("data_root is not in the dataframe and is not provided as an input")
            
        img_path = glob.glob(os.path.join(data_root, f"{row.index[0]}.jpg"))[0]
        # now copy the image to the distination_path directory and rename it to the index of the row
        if new_file_name is None:
            new_file_name = f"{row.index[0]}"
        shutil.copyfile(img_path, os.path.join(distination_path, f'{new_file_name}.jpg'))
    
def load_npz(data_table, data_root=None, filename=None, **kwargs):
    # find the row in the dataframe that match the input kwargs, the colums which are not mentioned in the kwargs are None
    row = data_table.loc[(data_table[list(kwargs)] == pd.Series(kwargs)).all(axis=1)]
    # if the row is empty return None
    if len(row) != 1:
        print(row)
        raise ValueError(f"the input kwargs: {kwargs} does not match any row in the dataframe or match more than one row")
    # if the row is not empty load the image
    else:
        if data_root is None:
            try:
                data_root = row["data_root"].values[0]
            except:
                raise ValueError("data_root is not in the dataframe and is not provided as an input")
          
        npz_path = glob.glob(os.path.join(data_root, f"{row.index[0]}.npz"))[0]
        npz = np.load(npz_path)
        if filename is None:
            return npz
        else:
            return npz[filename]

def get_df_row(data_table, **kwargs):
    return data_table.loc[(data_table[list(kwargs)] == pd.Series(kwargs)).all(axis=1)]

def column_inquary(col, data_table, **kwargs):
    # find the row in the dataframe that match the input kwargs, the colums which are not mentioned in the kwargs are None
    row = data_table.loc[(data_table[list(kwargs)] == pd.Series(kwargs)).all(axis=1)]
    # if the row is empty return None
    if len(row) != 1:
        raise ValueError(f"the input kwargs: {kwargs} does not match any row in the dataframe or match more than one row")
    # if the row is not empty load the image
    else:
        return row[col].values[0]


def l2_distance(tensor1, tensor2):
    return torch.sqrt(torch.sum((tensor1 - tensor2)**2))

def lpsips_distance(tensor1, tensor2, LPIPS_loss_fn):
    #convert the tensor from (3, 256, 256) to (1, 3, 256, 256)
    im0 = tensor1.unsqueeze(0)
    im1 = tensor2.unsqueeze(0)
    return LPIPS_loss_fn.forward(im0,im1)


# function to get two torch vector pearson correlation coefficient
def corrcoef_torch(score_tsr):
   return torch.corrcoef(score_tsr)[0, 1]

def get_similarty_score_by_CNN(img1, img2, scorerdist,  meta_data_df, row_idx, layer=None):
    score_list = list()
    if layer is not None:
        for modelname, scorer in scorerdist.items():
            imgtsr = torch.stack([img1, img2])
            model_output = scorer.model(imgtsr.to(torch.device('cuda:0'))).cpu().detach()
            # corolation score of two encoded images
            score_list.append(corrcoef_torch(model_output).item())
            meta_data_df.loc[row_idx, modelname] = corrcoef_torch(model_output).item()
    else:
        '''
        for modelname, scorer in scorerdist.items():
            if layer == "last_fc":
                if modelname == "alexnet":                   
                    layer_name = ".classifier.Linear4"
        '''
        pass 

def image_rf_crop(image, rf_size):
    preprocess = Compose([           
            CenterCrop(rf_size)])
    return preprocess(image)

def add_padding(tensor_image, padding_size, padding_color, colortpe="RGB1"):
    # Get the original image size
    original_size = tensor_image.size()

    # Calculate the new size after padding
    new_size = (original_size[1] + 2 * padding_size, original_size[2] + 2 * padding_size)

    # Create a new tensor with the new size and fill it with the padding color
    padded_tensor = torch.zeros(3, new_size[0], new_size[1], dtype=torch.float)
    if colortpe == "RGB255":
        padded_tensor[0, :, :] = padding_color[0] / 255.0  # Red channel
        padded_tensor[1, :, :] = padding_color[1] / 255.0  # Green channel
        padded_tensor[2, :, :] = padding_color[2] / 255.0  # Blue channel
    elif colortpe == "RGB1":
        padded_tensor[0, :, :] = padding_color[0]   # Red channel
        padded_tensor[1, :, :] = padding_color[1]   # Green channel
        padded_tensor[2, :, :] = padding_color[2]   # Blue channel
    else:
        raise ValueError("colortpe should be RGB255 or RGB1")

    # Insert the original image tensor into the padded tensor

    padded_tensor[:, padding_size:padding_size+original_size[1], padding_size:padding_size+original_size[2]] = tensor_image

    return padded_tensor

#def normalized_pixel_similarity(target_imag, reconstracted_imag, init_img, RFfilter=None):
#    delta_rec = (target_imag - reconstracted_imag).abs()
#    delta_init = (target_imag - init_img).abs()
#    similarity_idx_mat = delta_init - delta_rec 
#    if RFfilter is not None:
#        similarity_idx_mat[:, np.logical_not(RFfilter)] = 0
#        num_pixels = np.sum(RFfilter)
#    else:
#        num_pixels = np.prod(target_imag.shape[1:])
#    
#    return (similarity_idx_mat.sum()/num_pixels).item()

def sim_index_l1(target_imag, reconstracted_imag, init_img, RFfilter=None):
    if RFfilter is not None:
        target_imag[:, np.logical_not(RFfilter)] = 0
        reconstracted_imag[:, np.logical_not(RFfilter)] = 0
        init_img[:, np.logical_not(RFfilter)] = 0
        num_pixels = np.sum(RFfilter)
    else:
        num_pixels = np.prod(target_imag.shape[1:])
        raise Warning("RFfilter is None, it is kindda strange")
    
    delta_rec = (target_imag - reconstracted_imag).abs()
    delta_init = (target_imag - init_img).abs()
    similarity_idx_mat = delta_init - delta_rec
    
    return ((similarity_idx_mat.sum()/num_pixels)/ (delta_init.sum()/num_pixels)).item()

def sim_index_l2(target_imag, reconstracted_imag, init_img, RFfilter=None):
    if RFfilter is not None:
        target_imag[:, np.logical_not(RFfilter)] = 0
        reconstracted_imag[:, np.logical_not(RFfilter)] = 0
        init_img[:, np.logical_not(RFfilter)] = 0
    else:
        raise ValueError("RFfilter can not be None")
            
    delta_rec = (target_imag - reconstracted_imag).square().sum().sqrt()
    delta_init = (target_imag - init_img).square().sum().sqrt()
    return ((delta_init-delta_rec)/delta_init).item()

def normalized_l2(target_imag, reconstracted_imag, RFfilter=None):
    if RFfilter is not None:
        target_imag[:, np.logical_not(RFfilter)] = 0
        reconstracted_imag[:, np.logical_not(RFfilter)] = 0
        num_pixels = np.sum(RFfilter)
    else:
        raise ValueError("RFfilter can not be None")
            
    delta_rec = (target_imag - reconstracted_imag).square().sum().sqrt()/num_pixels
    return delta_rec.item()


def RF_mask_and_crop(imgs, rf_obj, RF_treshold=2):
    #RF_filter = rf_obj["fitmap"] > rf_obj["fitmap"][int(rf_obj["xo"]+(1.5*rf_obj["sigma_x"])), int(rf_obj["yo"]+(1.5*rf_obj["sigma_y"]))]
    RF_filter = rf_obj["fitmap"]
    imgs_masked =\
        (torch.from_numpy(np.absolute(RF_filter[None, None,:,:])) / RF_filter.max()) *\
        imgs
    imgs_masked_crooped = image_rf_crop(imgs_masked, (int(rf_obj["sigma_y"]*RF_treshold*2),
                                                     int(rf_obj["sigma_x"]*RF_treshold*2)))
    return imgs_masked_crooped