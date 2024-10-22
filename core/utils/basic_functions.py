import os
import numpy as np
import torch
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f
from PIL import ImageDraw
from torchvision.transforms import ToPILImage, PILToTensor, ToTensor
from PIL import Image
from torchvision import transforms
import mat73
from datetime import datetime




def extract_info_proto_exp(filename):
    # Define the regular expression pattern to match the filename format
    pattern = r'^(.*?)_thread(\d+)_expId(\d+)_chan(\d+)_unit(\d+)\.mat$'
    
    # Use the re.match function to find matches
    match = re.match(pattern, filename)
    
    # Check if the pattern was found within the filename
    if match:
        # Extract the matched groups
        exp_name, ithread, expId, chan, unit = match.groups()
        
        # Convert numerical values to integers
        ithread = int(ithread)
        expId = int(expId)
        chan = int(chan)
        unit = int(unit)
        
        # Return the extracted information as a dictionary
        return {
            'exp_name': exp_name,
            'ithread': ithread,
            'expId': expId,
            'chan': chan,
            'unit': unit
        }
    else:
        # Return None if the filename does not match the expected format
        return None


##------------
def process_gen_image_strings(cell_array):
    # Initialize arrays
    n = len(cell_array)
    is_gen = np.full(n, False)
    block_ids = np.full(n, np.nan)
    thread_ids = np.full(n, np.nan)
    gen_ids = np.full(n, np.nan)
    counters = np.full(n, np.nan)
    is_nat = np.full(n, False)
    nat_gen = np.full(n, np.nan)

    # Define the regular expression patterns
    gen_pattern = r'^block(\d{3})_thread(\d{3})_gen_gen(\d{3})_(\d{6})$'
    nat_pattern = r'_nat'
    last_gen_id = 0

    for i, string in enumerate(cell_array):
        tokens_gen = re.findall(gen_pattern, string)
        tokens_nat = nat_pattern in string

        # Check if the string matches the pattern
        if tokens_gen:
            is_gen[i] = True
            tokens_gen = tokens_gen[0]

            # Extract and convert each part of the string
            block_ids[i] = int(tokens_gen[0])
            thread_ids[i] = int(tokens_gen[1])
            gen_ids[i] = int(tokens_gen[2])
            counters[i] = int(tokens_gen[3])
            last_gen_id = gen_ids[i]

        # Check if the string matches the pattern for natural image
        if tokens_nat:
            is_nat[i] = True
            nat_gen[i] = last_gen_id

    return is_gen, block_ids, thread_ids, gen_ids, counters, is_nat, nat_gen


def analyze_by_gen(values, gen_ids, remove_last_gen=True):
    """
    Analyze by Generation - Calculates and sorts mean, standard deviation, and SEM for each generation
    for 2D input values, performing operations along the second dimension.

    Parameters:
    values (np.ndarray): 2D array of values. If 1D, it will be reshaped.
    gen_ids (np.ndarray): Array of corresponding generation IDs.
    remove_last_gen(bool): If True, the last generation will be removed from the output arrays becuse it is not complete

    Returns:
    gen_means (np.ndarray): Mean of values for each generation, sorted by generation ID.
    gen_stds (np.ndarray): Standard deviation of values for each generation, sorted by generation ID.
    gen_sems (np.ndarray): Standard error of the mean for each generation, sorted by generation ID.
    unique_gens (np.ndarray): Sorted list of unique generation IDs.
    """
    # Ensure values are 2D, reshaping if necessary
    if values.ndim == 1:
        values = values.reshape(1, -1) #TODO: check if it is correct

    # Find unique generation IDs and sort them
    unique_gens = np.unique(gen_ids)
    n_gens = len(unique_gens)

    # Initialize output arrays
    gen_means = np.zeros((values.shape[0], n_gens))
    gen_stds = np.zeros((values.shape[0], n_gens))
    gen_sems = np.zeros((values.shape[0], n_gens))

    # Calculate statistics for each generation
    for i, gen in enumerate(unique_gens):
        gen_mask = gen_ids == gen
        gen_values = values[:, gen_mask]

        gen_means[:, i] = np.nanmean(gen_values, axis=1)
        gen_stds[:, i] = np.nanstd(gen_values, axis=1)
        n_values = np.sum(gen_mask)  # Number of values for this generation
        gen_sems[:, i] = gen_stds[:, i] / np.sqrt(n_values)

    if remove_last_gen:
        gen_means = gen_means[:, :-1]
        gen_stds = gen_stds[:, :-1]
        gen_sems = gen_sems[:, :-1]
        unique_gens = unique_gens[:-1]

    if values.ndim == 1:
        gen_means = gen_means.squeeze()
        gen_stds = gen_stds.squeeze()
        gen_sems = gen_sems.squeeze()

    return gen_means, gen_stds, gen_sems, unique_gens

#-------------
def get_score_from_resp(gen_resp_array, target_resp_vec, mask_vec, score_mode, popu_mean, popu_std, diff_mask = False, second_mask = None):
    """
    Calculates the score based on the given mode.

    This function normalizes the response arrays and computes a score
    based on the specified scoring mode. Currently, it supports Mean
    Squared Error (MSE).

    Parameters:
    gen_resp_array : numpy.ndarray
        The generated response array.
    target_resp_vec : numpy.ndarray
        The target response vector.
    mask_vec : numpy.ndarray
        A vector used for masking elements in the calculation.
    score_mode : str
        The mode of scoring ('MSE', etc.)
    popu_mean : float
        The population mean used for normalization.
    popu_std : float
        The population standard deviation used for normalization.

    diff_mask : bool: default False
        If True, it will use the second mask to mask the mean and std of the population
        It's happen if we use the resopnse that comes from the selectivity experiment
    second_mask : np.ndarray: default None
        The second mask to be used for masking the mean and std of the population
    Returns:
    numpy.ndarray
        The calculated score.

    """

   # Normalize the response
    gen_resp_array_masked = gen_resp_array[mask_vec, :]
    target_resp_masked = target_resp_vec[mask_vec]
    if diff_mask:
        popu_mean_masked = popu_mean[second_mask]
        popu_std_masked = popu_std[second_mask]
    else:
        popu_mean_masked = popu_mean[mask_vec]
        popu_std_masked = popu_std[mask_vec]

    gen_resp_array_norm_masked = (gen_resp_array_masked - popu_mean_masked[:, None]) / popu_std_masked[:, None]
    target_resp_vec_norm_masked= (target_resp_masked - popu_mean_masked) / popu_std_masked

    # Compute score based on the specified mode
    if score_mode == 'MSE':
        score_vec = -np.nanmean((gen_resp_array_norm_masked - target_resp_vec_norm_masked[:, None]) ** 2, axis=0)
    elif score_mode == 'MSE-notNorm':
        score_vec = -np.nanmean((gen_resp_array_masked - target_resp_masked[:, None]) ** 2, axis=0)
    elif score_mode == 'cosine':
        # Compute the cosine similarity and convert it to cosine distance
        dot_product = np.sum(gen_resp_array_norm_masked * target_resp_vec_norm_masked[:, None], axis=0)
        norm_a = np.linalg.norm(gen_resp_array_norm_masked, axis=0)
        norm_b = np.linalg.norm(target_resp_vec_norm_masked)
        cosine_similarity = dot_product / norm_a / norm_b
        score_vec = cosine_similarity
    else:
        score_vec = -np.nanmean((gen_resp_array_norm_masked - target_resp_vec_norm_masked[:, None]) ** 2, axis=0)
        Warning((f"{score_mode} score mode is not yet implemented will use MSE instead"))
        

    return score_vec
#----------------
load_preprocess = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return load_preprocess(image)

from pathlib import Path

def list_image_files(directory):
    """
    Lists full paths of all image files in the given directory and its subdirectories.

    Args:
    directory (Path or str): The path to the directory.

    Returns:
    list: A list of full paths to image files.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif']  # Add other file types if needed
    directory = Path(directory)
    image_paths = [str(file) for file in directory.rglob('*') if file.suffix.lower() in image_extensions]
    
    return image_paths


def load_image_by_name(image_name, image_dir, image_list=None):
    if image_list is None:
        image_paths = list_image_files(image_dir)
    else:
        image_paths = image_list
    image_path = [x for x in image_paths if image_name == '.'.join(x.split('\\')[-1].split('.')[:-1])]
    if len(image_path) == 0:
        return None
    return load_image(image_path[0])

def load_all_images(image_dir, image_list=None):
    if image_list is None:
        image_paths = list_image_files(image_dir)
    else:
        image_paths = image_list
    img_tensor = list()
    image_names_list = list()
    for img_path in image_paths:
        img = load_image(img_path)
        img_tensor.append(img)
        # let get image name without the extension
        img_name = '.'.join(img_path.split('\\')[-1].split('.')[:-1])
        image_names_list.append(img_name)

    return torch.stack(img_tensor), np.array(image_names_list)


def load_image_by_gen_thread_id(image_dir, img_name_array, score_vec, gen_id, thread_id, resp_mat, k=5, random_seed=42):
    # return the images for the coresponded gen and thread id, it will return the best image, mean image, or top k mean image
    # let get the image name
    pattern = f'thread{thread_id:03d}_gen_gen{gen_id:03d}'
    gen_matching_indices = [i for i, s in enumerate(img_name_array) if pattern in s]
    image_paths = list_image_files(image_dir)
    gen_scores = score_vec[gen_matching_indices]
    gen_resp = resp_mat[:, gen_matching_indices]
    gen_image_names = img_name_array[gen_matching_indices]

    img_tensor = list()
    for img_name in gen_image_names:
        img = load_image_by_name(img_name, image_dir, image_paths)
        img_tensor.append(img)
    img_tensor = torch.stack(img_tensor)
    #  function will return a dictionary that contains folowing:
    # mean_image, mean_score, best_image, best_score, top_k_images, top_k_score, random_images, random_score, mean_top_k_images, mean_top_k_score,
    # mean_random_images, mean_random_score
    # let get the mean image
    mean_img = torch.mean(img_tensor, dim=0).squeeze()
    mean_score = np.mean(gen_scores)

    # get randon k images forn the tensor with the same seed
    # seeting the seed
    np.random.seed(random_seed)
    random_indices = np.random.choice(range(len(gen_scores)), k, replace=False)
    
    random_images = img_tensor[random_indices.copy()]
    random_images_score = gen_scores[random_indices]
    random_images_resp = gen_resp[:, random_indices]

    random_images_mean = torch.mean(random_images, dim=0).squeeze()
    random_images_mean_score = np.mean(random_images_score)
    random_images_mean_resp = np.mean(random_images_resp, axis=1)

    # let get the best image
    bset_img_idx = np.argmax(gen_scores)
    best_img = img_tensor[bset_img_idx].squeeze()
    best_score = gen_scores[bset_img_idx]
    best_resp = gen_resp[:, bset_img_idx]

    # get the top k images
    top_k_indices = np.argsort(gen_scores)[::-1][:k]
    top_k_images = img_tensor[top_k_indices.copy()]
    top_k_score = gen_scores[top_k_indices]
    top_k_resp = gen_resp[:, top_k_indices]

    top_k_imgs_mean = torch.mean(top_k_images, dim=0).squeeze()
    top_k_score_mean = np.mean(top_k_score)
    top_k_resp_mean = np.mean(top_k_resp, axis=1)

    # let make the dictionary
    image_score_dict = {
        'mean_image': mean_img,
        'mean_score': mean_score,
        'best_image': best_img,
        'best_score': best_score,
        'top_k_images': top_k_images,
        'top_k_score': top_k_score,
        'random_images': random_images,
        'random_score': random_images_score,
        'mean_top_k_images': top_k_imgs_mean,
        'mean_top_k_score': top_k_score_mean,
        'mean_random_images': random_images_mean,
        'mean_random_score': random_images_mean_score,
        'best_resp': best_resp,
        'top_k_resp': top_k_resp,
        'random_images_resp': random_images_resp,
        'top_k_resp_mean': top_k_resp_mean,
        'random_images_mean_resp': random_images_mean_resp
    }

    return image_score_dict

#--------------

def make_binary_mask_np(mask, x=1.25):
    """
    Converts a mask with continuous values between 0 and 1 into a binary mask.
    The binary threshold is set to mean + x * standard deviation of the mask values.
    
    Parameters:
    mask (np.ndarray): 2D array of continuous values between 0 and 1.
    x (float): Multiplier for the standard deviation to define the threshold.
    
    Returns:
    binary_mask (np.ndarray): 2D binary mask with the same shape as the input mask.
    """
    # Calculate mean and standard deviation of the mask values
    mean_val = np.mean(mask)
    std_val = np.std(mask)
    
    # Define the threshold as mean + x * standard deviation
    threshold = mean_val + x * std_val
    
    # Generate binary mask: 1 for values greater than the threshold, 0 otherwise
    binary_mask = np.where(mask > threshold, 1, 0)
    
    return binary_mask

def make_binary_mask_torch(mask, x_std=1):
    """
    Converts a mask with continuous values between 0 and 1 into a binary mask using PyTorch.
    The binary threshold is set to mean + x * standard deviation of the mask values.
    
    Parameters:
    mask (torch.Tensor): 2D tensor of continuous values between 0 and 1.
    x (float): Multiplier for the standard deviation to define the threshold.
    
    Returns:
    binary_mask (torch.Tensor): 2D binary mask with the same shape as the input mask.
    """
    # Ensure mask is a float tensor for mean and std calculations
    mask_float = mask.float()
    
    # Calculate mean and standard deviation of the mask values
    mean_val = torch.mean(mask_float)
    std_val = torch.std(mask_float)
    
    # Define the threshold as mean + x * standard deviation
    threshold = mean_val + x_std * std_val
    
    # Generate binary mask: 1 for values greater than the threshold, 0 otherwise
    binary_mask = (mask_float > threshold).float()  # Convert to float for 0.0 and 1.0 values
    
    return binary_mask

#-------------
def average_nonzero(tensor, dims):
    """
    Calculates the average of all non-zero values in the specified dimensions of a PyTorch tensor.
    
    Parameters:
    tensor (torch.Tensor): The input tensor.
    dims (tuple): The dimensions along which to calculate the average of non-zero values.
    
    Returns:
    avg_nonzero (torch.Tensor): The average of non-zero values along the specified dimensions.
    """
    # Create a mask of non-zero values
    nonzero_mask = tensor != 0

    # let return zero if all values are zero
    if torch.sum(nonzero_mask) == 0:
        return torch.tensor(0.0, device=tensor.device)

    # Use the mask to select non-zero values
    nonzero_values = tensor * nonzero_mask
    
    # Calculate the sum and the count of non-zero values along the specified dimensions
    nonzero_sum = torch.sum(nonzero_values, dim=dims, keepdim=True)
    nonzero_count = torch.sum(nonzero_mask, dim=dims, keepdim=True)

    # Avoid division by zero by using where to only calculate average where count > 0
    avg_nonzero = torch.where(nonzero_count > 0, nonzero_sum / nonzero_count, torch.tensor(0.0, device=tensor.device))
    
    return avg_nonzero.squeeze() # Remove the squeezed dimensions if you want to

#---------------
def get_lreg_sklearn_all_units(resp, gen_id, zs_flag = 1, abs_flag = 1):
    """
    Applies optional absolute value, z-scores responses, removes NaNs, and performs linear regression on each row (unit),
    initializing output arrays with np.nan.
    """

    # Z-score responses for each unit along columns
    if zs_flag == 1:
        resp_zs = (resp - np.nanmean(resp, axis=1, keepdims=True)) / np.nanstd(resp, axis=1, keepdims=True)
    else:
        resp_zs = resp

    # Apply absolute value if flagged
    if abs_flag == 1:
        resp_zs = np.abs(resp_zs)

    # Initialize output arrays with np.nan
    n_units = resp_zs.shape[0]
    slope_vec = np.full(n_units, np.nan)
    intercept_vec = np.full(n_units, np.nan)
    pValue_vec = np.full(n_units, np.nan)
    r2_vec = np.full(n_units, np.nan)  # Initialize R² array

   # Instantiate LinearRegression object outside the loop
    regr = LinearRegression()

    for ui in range(n_units):
        # Identify rows without NaNs
        valid_indices = ~np.isnan(resp_zs[ui, :])
        valid_resp = resp_zs[ui, valid_indices]
        valid_gen_id = gen_id[valid_indices].reshape(-1, 1)

        if valid_resp.size > 0:  # Proceed if there are non-NaN values
            # Fit linear regression model
            regr.fit(valid_gen_id, valid_resp)

            # Get coefficients
            intercept_vec[ui] = regr.intercept_
            slope_vec[ui] = regr.coef_[0]

            # Calculate R² value
            r2 = regr.score(valid_gen_id, valid_resp)
            r2_vec[ui] = r2

            # Calculate F-statistic and corresponding p-value
            n = len(valid_resp)
            p = 1  # Number of predictors, not counting the intercept
            dfn = p  # Degrees of freedom for the numerator
            dfd = n - p - 1  # Degrees of freedom for the denominator
            F = (r2 / dfn) / ((1 - r2) / dfd)
            pValue_vec[ui] = 1 - f.cdf(F, dfn, dfd)

    return slope_vec, intercept_vec, pValue_vec, r2_vec

#-----------------
def clean_data(x, y):
    valid_indices = ~np.isnan(x) & ~np.isnan(y)
    return x[valid_indices], y[valid_indices]
    
#-------------------
    
def list_mat_files(directory):
    # List all files in the directory
    all_files = os.listdir(directory)
    # Filter out files that end with '.mat'
    mat_files = [file for file in all_files if file.endswith('.mat')]
    return mat_files

def prune_path(in_path):
    # Split the path into parts
    path_split = in_path.split('\\')
    
    # Find the index where 'Stimuli' appears
    try:
        find_id = path_split.index('Stimuli')
    except ValueError:
        return None  # 'Stimuli' not found in the path
    
    # Construct the output path from the 'Stimuli' part onwards
    out_path = '\\'.join(path_split[find_id:])
    
    return out_path


# let's find the center of the mask 
def find_mask_center(mask, pixels_per_deg=128):
    mask = mask.numpy()
    mask_center = np.array(np.where(mask == 1))
    # get center by median of the x and y not the mean
    mask_center = np.median(mask_center, axis=1)
    # let find this center in cordinate cnetred at the center of the image
    mask_center_img_cordinte = np.array([mask_center[1] - mask.shape[1] / 2, mask.shape[0] / 2 - mask_center[0]])
    # convert to visual angle
    mask_center_angle = mask_center_img_cordinte / pixels_per_deg
    return mask_center, mask_center_img_cordinte, mask_center_angle


def add_text_to_image(img, text, position, font_size=20, font_color=(255, 0, 0)):
    if len(img.shape) == 3:
        img = ToPILImage()(img)
        draw = ImageDraw.Draw(img)
        draw.text(position, text, font_color)
        img = ToTensor()(img)
    if len(img.shape) == 4:
        img_list = list()
        for i in range(len(img)):
            img_ = ToPILImage()(img[i].squeeze())
            draw = ImageDraw.Draw(img_)
            draw.text(position, text[i], font_color)
            img_list.append(ToTensor()(img_))
        img = torch.stack(img_list)  
    return img


def load_mat_data(file_path):
    data = mat73.loadmat(file_path)
    for key in data.keys():
        if type(data[key]) == list:
            list_keys = data[key]
            if len(list_keys) == 1:
                list_keys = list_keys
            elif len(list_keys) > 1:
                list_keys = [x[0] for x in list_keys]
            else:
                raise ValueError('The list is empty')
            data[key] = np.array(list_keys)
    return data


def get_target_noise_celling(resp_map, mask, score_mods, pop_mean, pop_std, diff_mask = False, second_mask = None):
    # let get celling trial by trial
    all_trails_noise = list()
    for i in range(resp_map.shape[1]):
        target_resp = resp_map[:, i]
        score_vec = get_score_from_resp(resp_map, target_resp, mask, score_mods, pop_mean, pop_std, diff_mask, second_mask)
        all_trails_noise.append(score_vec)
    
    # clling by random half draw of the trials
    num_of_draws =  50
    half_draw_noise = list()
    for i in range(num_of_draws):
        selected_idx = np.random.choice(range(resp_map.shape[1]), resp_map.shape[1]//2, replace=False)
        not_selected_idx = np.setdiff1d(range(resp_map.shape[1]), selected_idx)
        first_half_resp = np.mean(resp_map[:, selected_idx], axis=1, keepdims=True)
        second_half_resp = np.mean(resp_map[:, not_selected_idx], axis=1)
        score_vec_selected = get_score_from_resp(first_half_resp, second_half_resp, mask, score_mods, pop_mean, pop_std, diff_mask, second_mask)
        half_draw_noise.append(score_vec_selected[0])
    
    # celling by one out of the trials
    one_out_noise = list()
    for i in range(resp_map.shape[1]):
        trial_idx = i
        not_trial_idx = np.setdiff1d(range(resp_map.shape[1]), trial_idx)
        trial_resp = resp_map[:, trial_idx]
        not_trial_resp = np.mean(resp_map[:, not_trial_idx], axis=1, keepdims=True)
        score_vec_one_out = get_score_from_resp(not_trial_resp, trial_resp, mask, score_mods, pop_mean, pop_std, diff_mask, second_mask)
        one_out_noise.append(score_vec_one_out[0])
    
    return all_trails_noise, half_draw_noise, one_out_noise


def get_selectivity_exp_coresponded_mask(recon_data, iThread):
    # its a super local function for recon expriment analysis
    
    unit_ids_recon_masked = recon_data['unit_ids_recon'][recon_data['masks_cell'][iThread]]
    chan_ids_recon_masked = recon_data['chan_ids_recon'][recon_data['masks_cell'][iThread]]
    chan_unit_recon_masked = np.stack([chan_ids_recon_masked, unit_ids_recon_masked], axis=1)
    chan_unit_recon_masked = [tuple(x) for x in chan_unit_recon_masked]

    # let get the mask for the response from the selectivity
    unit_ids_select = recon_data['unit_ids_select']
    chan_ids_select = recon_data['chan_ids_select']
    chan_unit_select = np.stack([chan_ids_select, unit_ids_select], axis=1)
    chan_unit_select = [tuple(x) for x in chan_unit_select]
    mask_celect = np.array([x in chan_unit_recon_masked for x in chan_unit_select])
    
    return mask_celect


def parse_date(date_str, animal_name='Caos'):
    if animal_name == 'Caos':
        # the date format is ambiguous, it can be DDMMYYYY or MMDDYYYY
        # Define the transition date
        transition_date = datetime(2023, 12, 6)
        lower_bound = datetime(2023, 11, 1)

        # Check the length of the string to determine the likely format
        if len(date_str) == 8:  # Both DDMMYYYY and MMDDYYYY are 8 digits long
            # Try parsing as DDMMYYYY
            try:
                parsed_date = datetime.strptime(date_str, '%d%m%Y')
                if (parsed_date < transition_date) & (parsed_date >= lower_bound):
                    return parsed_date
            except ValueError:
                pass

            # Try parsing as MMDDYYYY
            try:
                parsed_date = datetime.strptime(date_str, '%m%d%Y')
                if (parsed_date >= transition_date)  & (parsed_date >= lower_bound):
                    return parsed_date
            except ValueError:
                pass
        elif len(date_str) in [7, 6]:  # Handling cases for MDDYYYY or MMDDYY
            formats = ['%m%d%Y', '%d%m%Y'] if len(date_str) == 7 else ['%m%d%y', '%d%m%y']
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    if parsed_date >= transition_date:
                        return parsed_date
                    elif parsed_date.year < 100:  # Adjust for two-digit year format interpretation
                        parsed_date = parsed_date.replace(year=parsed_date.year + 2000)
                        if parsed_date < transition_date:
                            return parsed_date
                except ValueError:
                    continue

        # If no valid date format is found
        raise ValueError("Invalid date format or ambiguous date")
    elif animal_name == 'Diablito':
        # here all the dates are in the format of DDMMYYYY
        return datetime.strptime(date_str, '%d%m%Y')
    else:
        raise ValueError("The animal name is not supported")


def plot_madoul(x_clean, y_clean, ax):
    ax.scatter(x_clean, y_clean)
    # Line of best fit
    m1, b1 = np.polyfit(x_clean, y_clean, 1)
    ax.plot(x_clean, m1*x_clean + b1, color='red')  # Add line of best fit
    # Calculate the correlation coefficient
    corr_coeff1 = np.corrcoef(x_clean, y_clean)[0, 1]
    ax.text(.7, 0.95, f'Corr Coeff: {corr_coeff1:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top')


def clean_data(x, y):
    # Remove NaNs and inf
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
    x_clean = x[mask]
    y_clean = y[mask]