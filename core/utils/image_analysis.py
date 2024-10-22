from core.utils.basic_functions import *
import numpy as np
import torch
# cv2
import cv2
import lpips  # Make sure you have the LPIPS library installed


def get_real_pos_img_unit8(img, center_pos_deg, img_size_deg, screen_size=3, grayness_level=0.5, pixels_per_deg=128):
    """
    Places an image on an imaginary screen with specified parameters.

    Parameters:
    - img: Input image as a NumPy array.
    - center_pos_deg: Tuple or list with 2 elements indicating the center position of the image in degrees.
    - img_size_deg: Size of the image in degrees.
    - screen_size: Size of the screen in degrees. Default is 3.
    - grayness_level: Grayness level of the screen, between 0 and 1. Default is 0.5.
    - pixels_per_deg: Number of pixels per degree. Default is 128.

    Returns:
    - A NumPy array representing the output image with the input image placed on the gray screen.
    """

    # Constants
    screen_size_pixels = int(screen_size * pixels_per_deg)

    # Create a gray background screen
    # make image if the input image! (i.e., 3D)
    if len(img.shape) == 3:
        out_img = np.uint8(255 * grayness_level * np.ones((screen_size_pixels, screen_size_pixels, 3)))
    elif len(img.shape) == 2:
        out_img = np.zeros((screen_size_pixels, screen_size_pixels))
    else:
        raise ValueError("Input image must be either 2D (grayscale) or 3D (RGB).")

    # Calculate the image size in pixels
    img_size_pixels = int(img_size_deg * pixels_per_deg)
    center_pos_deg_rev = [-center_pos_deg[1], center_pos_deg[0]]
    
    # Calculate the top-left corner of the image based on the center position
    center_pos_pixels = np.array(center_pos_deg_rev) * pixels_per_deg + np.array([screen_size_pixels / 2, screen_size_pixels / 2])
    top_left_corner = np.round(center_pos_pixels - img_size_pixels / 2).astype(int)
    
    # Resize the input image to the specified size in pixels
    resized_img = cv2.resize(img, (img_size_pixels, img_size_pixels))

    # Determine the cropping bounds
    crop_row_start, crop_col_start = max(1, -top_left_corner[0] + 1), max(1, -top_left_corner[1] + 1)
    crop_row_end, crop_col_end = min(img_size_pixels, screen_size_pixels - top_left_corner[0]), min(img_size_pixels, screen_size_pixels - top_left_corner[1])

    # Crop the image if it goes outside the screen
    if len(img.shape) == 3:
        cropped_img = resized_img[crop_row_start-1:crop_row_end, crop_col_start-1:crop_col_end, :]
    elif len(img.shape) == 2:
        cropped_img = resized_img[crop_row_start-1:crop_row_end, crop_col_start-1:crop_col_end]

    # Update the top-left corner for placing the cropped image
    top_left_corner = np.maximum(top_left_corner, [1, 1])
    bottom_right_corner = np.minimum(top_left_corner + np.array(cropped_img.shape[:2]) - 1, [screen_size_pixels, screen_size_pixels])

    # Place the cropped/resized image on the gray screen
    if len(img.shape) == 3:
        out_img[top_left_corner[0]-1:bottom_right_corner[0], top_left_corner[1]-1:bottom_right_corner[1], :] = cropped_img
    elif len(img.shape) == 2:
        out_img[top_left_corner[0]-1:bottom_right_corner[0], top_left_corner[1]-1:bottom_right_corner[1]] = cropped_img

    return out_img


import torchvision.transforms.functional as F

def get_real_pos_img_torch(img, center_pos_deg, img_size_deg, screen_size=3, grayness_level=0.5, pixels_per_deg=128):
    """
    Places an image on an imaginary screen with specified parameters, using PyTorch tensors.

    Parameters:
    - img (torch.Tensor): Input image as a PyTorch tensor of shape (C, H, W) or (1, H, W) for grayscale.
    - center_pos_deg (tuple or list): Center position of the image in degrees.
    - img_size_deg (float): Size of the image in degrees.
    - screen_size (float): Size of the screen in degrees. Default is 3.
    - grayness_level (float): Grayness level of the screen, between 0 and 1. Default is 0.5.
    - pixels_per_deg (int): Number of pixels per degree. Default is 128.

    Returns:
    - torch.Tensor: Output image with the input image placed on the gray screen.
    """

    # Constants
    screen_size_pixels = int(screen_size * pixels_per_deg)
    img_size_pixels = int(img_size_deg * pixels_per_deg)

    # Create a gray background screen
    if img.dim() == 3:
        out_img =  grayness_level * torch.ones((3, screen_size_pixels, screen_size_pixels))
    elif img.dim() == 2:
        out_img = grayness_level * torch.ones((1, screen_size_pixels, screen_size_pixels))
        img = img.unsqueeze(0)
    else:
        raise ValueError("Input image tensor must be 2D (1, H, W) for grayscale or 3D (C, H, W) for RGB.")

    center_pos_deg_rev = [-center_pos_deg[1], center_pos_deg[0]]
    
    # Calculate the top-left corner of the image based on the center position
    center_pos_pixels = np.array(center_pos_deg_rev) * pixels_per_deg + np.array([screen_size_pixels / 2, screen_size_pixels / 2])
    top_left_corner = np.round(center_pos_pixels - img_size_pixels / 2).astype(int)
    
    # Resize the input image to the specified size in pixels
    resized_img = F.resize(img, [img_size_pixels, img_size_pixels])

    # Determine the cropping bounds
    crop_row_start, crop_col_start = max(1, -top_left_corner[0] + 1), max(1, -top_left_corner[1] + 1)
    crop_row_end, crop_col_end = min(img_size_pixels, screen_size_pixels - top_left_corner[0]), min(img_size_pixels, screen_size_pixels - top_left_corner[1])

    # Crop the image if it goes outside the screen
    cropped_img = resized_img[:, crop_row_start-1:crop_row_end, crop_col_start-1:crop_col_end]

    # Update the top-left corner for placing the cropped image
    top_left_corner = np.maximum(top_left_corner, [1, 1])
    bottom_right_corner = np.minimum(top_left_corner + np.array(cropped_img.shape[1:]) - 1, [screen_size_pixels, screen_size_pixels])

    # Place the cropped/resized image on the gray screen
    out_img[:, top_left_corner[0]-1:bottom_right_corner[0], top_left_corner[1]-1:bottom_right_corner[1]] = cropped_img

    return out_img.squeeze()

#---------------------

def np_image_to_torch_tensor(np_image):
    """
    Convert a NumPy image array (H, W, C) with dtype=uint8 to a PyTorch tensor (C, H, W) with float values scaled relative to the maximum value in the tensor.

    Parameters:
    np_image (numpy.ndarray): Input image as a NumPy array with shape (H, W, C) and dtype=uint8.

    Returns:
    torch.Tensor: Output image as a PyTorch tensor with shape (C, H, W) and values scaled relative to the maximum value.
    """
    # Ensure input is a NumPy array
    if not isinstance(np_image, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    # Check if the input array has 3 dimensions (H, W, C)
    if np_image.ndim != 3:
        raise ValueError("Input array must have 3 dimensions (H, W, C)")

    # Convert to (C, H, W)
    np_image_transposed = np.transpose(np_image, (2, 0, 1))

    # Convert to a PyTorch tensor
    tensor_image = torch.from_numpy(np_image_transposed).float()

    # Scale relative to the maximum value
    max_value = torch.max(tensor_image)
    if max_value > 0:  # Avoid division by zero
        tensor_image /= max_value

    return tensor_image
## ---------------

def apply_mask_to_image_uint8(image, mask):
    """
    Applies a given mask to a given image by expanding the mask to match the image's dimensions and then
    performing an element-wise multiplication.

    Parameters:
    - image: A numpy array representing the image, expected to be in the shape (height, width, channels).
    - mask: A numpy array representing the mask, expected to be in the shape (height, width) with values between 0 and 1.

    Returns:
    - A numpy array of the masked image, with the same shape as the input image.
    """

    # Expand the mask dimensions to match the image's channels
    expanded_mask = np.expand_dims(mask, axis=-1)  # Add a third dimension to the mask
    expanded_mask = np.repeat(expanded_mask, image.shape[2], axis=-1)  # Repeat the mask for each channel

    # Multiply the image with the expanded mask and convert to uint8
    masked_image = (image * expanded_mask).astype('uint8')

    return masked_image


def apply_mask_to_image_torch(image, mask):
    """
    Applies a given mask to a given image batch using PyTorch. The mask is expanded to match the image's dimensions and then
    an element-wise multiplication is performed.

    Parameters:
    - image (torch.Tensor): A tensor representing the image batch, expected to be in the shape (B, C, H, W).
    - mask (torch.Tensor): A tensor representing the mask batch, expected to be in the shape (B, H, W) with values between 0 and 1.

    Returns:
    - torch.Tensor: A tensor of the masked image batch, with the same shape as the input images.
    """

    # Ensure tensors are in batch form
    if image.dim() == 3:  # If the image tensor is (C, H, W), add the batch dimension
        image = image.unsqueeze(0)
    if mask.dim() == 2:  # If the mask tensor is (H, W), add the batch dimension
        mask = mask.unsqueeze(0)

    # Add the channel dimension to the mask (B, 1, H, W)
    mask = mask.unsqueeze(1)

    # Repeat the mask for each channel in the image
    mask = mask.expand(-1, image.size(1), -1, -1)  # Expand the mask to match the image's channels

    # Multiply the image with the expanded mask
    masked_image = image * mask

    return masked_image

#-------------------
def compute_lpips_similarity_unit8(img1_np, img2_np, net_type='alex'): # TODO: make it bach freindly
    """
    Compute the LPIPS similarity between two images.

    Parameters:
    img1_np (numpy.ndarray): First input image as a NumPy array with shape (H, W, C) and dtype=uint8.
    img2_np (numpy.ndarray): Second input image as a NumPy array with shape (H, W, C) and dtype=uint8.
    net_type (str): Type of network to use for LPIPS computation ('alex', 'vgg', etc.).

    Returns:
    float: LPIPS similarity value.
    """
    # Initialize the LPIPS model
    #lpips_model = lpips.LPIPS(net=net_type).cuda() if torch.cuda.is_available() else lpips.LPIPS(net=net_type)
    lpips_model = lpips.LPIPS(net=net_type, spatial=True)

    
    # Convert NumPy arrays to PyTorch tensors and add batch dimension
    img1_tensor = np_image_to_torch_tensor(img1_np).unsqueeze(0)  # Convert and add batch dimension
    img2_tensor = np_image_to_torch_tensor(img2_np).unsqueeze(0)  # Convert and add batch dimension

    # Compute LPIPS similarity
    with torch.no_grad():
        similarity_map = lpips_model.forward(img1_tensor, img2_tensor)
    
    # Convert similarity map to image for visualization
    similarity_map_img = similarity_map.squeeze().cpu().numpy()  # Remove batch dimension and transfer to CPU
    similarity_map_img = (similarity_map_img - similarity_map_img.min()) / (similarity_map_img.max() - similarity_map_img.min())  # Normalize to 0-1 range

    return similarity_map, similarity_map_img


def uint8_image_to_pytorch(image):
    """
    Convert a uint8 image (numpy array) to a PyTorch tensor.

    Parameters:
        image (numpy.ndarray): An image array with shape (H, W, C) and uint8 type.

    Returns:
        torch.Tensor: A PyTorch tensor of the image with shape (C, H, W) and float type.
    """
    # Convert the image to float and scale to [0, 1]
    image_float = image.astype(np.float32) / 255.0
    
    # Rearrange the dimensions from HWC to CHW if the image is color
    if image_float.ndim == 3:
        image_transposed = np.transpose(image_float, (2, 0, 1))
    else:
        image_transposed = image_float
        # add a new C axis to the image
        image_transposed = np.expand_dims(image_transposed, axis=0)
    
    # Convert the NumPy array to a PyTorch tensor
    image_tensor = torch.from_numpy(image_transposed)
    
    return image_tensor

def compute_lpips_similarity(img1, img2_batch, net_type='alex'):
    """
    Compute the LPIPS similarity between one image and a batch of images.

    Parameters:
    - img1 (torch.Tensor): First input image as a torch tensor with shape (C, H, W).
    - img2_batch (torch.Tensor): Batch of second input images as a torch tensor with shape (B, C, H, W).
    - net_type (str): Type of network to use for LPIPS computation ('alex', 'vgg', etc.).

    Returns:
    - torch.Tensor: LPIPS similarity values for each image in the batch.
    """

    # Initialize the LPIPS model with the specified network type
    lpips_model = lpips.LPIPS(net=net_type, spatial=True)  # spatial=True returns a spatial map of similarities

    # Add an extra batch dimension to img1 to match the (N, C, H, W) format
    img1_batch = img1.unsqueeze(0)

    # Move the LPIPS model to the same device as the input tensors
    if torch.cuda.is_available():
        lpips_model = lpips_model.to('cpu')
        img1 = img1.to('cpu')
        img2_batch = img2_batch.to('cpu')
    else:
        device = img1_batch.device  # Get the device from the input tensor
        img2_batch = img2_batch.to(device)  # Move the batch of images to the correct device
        lpips_model = lpips_model.to(device)  # Move LPIPS model to the correct device

    # Compute LPIPS similarity
    similarity_map = lpips_model.forward(img1_batch, img2_batch)

    # Convert similarity map to image for visualization
    # Normalize the similarity map to be between 0 and 1 for visualization
    similarity_map_norm = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min())
    #similarity_map_img = similarity_map_norm.squeeze().cpu().detach()  # Remove batch dimension and move to CPU

    return similarity_map, similarity_map_norm

def compute_l2_distance(img1_batch, img2_batch):
    """
    Compute the L2 (Euclidean) distance between one image and a batch of images.

    Parameters:
    - img1 (torch.Tensor): First input image as a torch tensor with shape (C, H, W) or Batch of first input images as a torch tensor with shape (B, C, H, W).
    - img2_batch (torch.Tensor): Batch of second input images as a torch tensor with shape (B, C, H, W).

    Returns:
    - torch.Tensor: L2 distance values for each image in the batch.
    """

    # Add an extra batch dimension to img1 to match the (B, C, H, W) format
    if img1_batch.dim() == 3:
        img1_batch = img1_batch.unsqueeze(0)
    elif img2_batch.dim() == 3:
        img2_batch = img2_batch.unsqueeze(0)

    

    # Compute the L2 distance
    # This is done by taking the square root of the sum of the squared differences
    squared_diff = (img1_batch - img2_batch) ** 2

    l2_distance = torch.sqrt(((img1_batch - img2_batch) ** 2).sum(dim=[1, 2, 3]))

    return l2_distance

def compute_mse_map(img1_batch, img2_batch):
    """
    Compute the Mean Squared Error (MSE) map between one image and a batch of images. 
    The MSE map provides a spatial representation of the squared differences.

    Parameters:
    - img1 (torch.Tensor): First input image as a torch tensor with shape (C, H, W) or Batch of first input images as a torch tensor with shape (B, C, H, W).
    - img2_batch (torch.Tensor): Batch of second input images as a torch tensor with shape (B, C, H, W).

    Returns:
    - torch.Tensor: A batch of MSE maps with shape (B, H, W), representing the MSE at each spatial location.
    """

    # Add an extra batch dimension to img1 to match the (B, C, H, W) format
    if img1_batch.dim() == 3:
        img1_batch = img1_batch.unsqueeze(0)
    if img2_batch.dim() == 3:
        img2_batch = img2_batch.unsqueeze(0)


    # Compute the squared differences
    squared_diff = (img1_batch - img2_batch) ** 2

    mse_map = average_nonzero(squared_diff, dims=[1])


    # Compute the MSE map by averaging the squared differences across the channel dimension
    return mse_map

def compute_mse(img1_batch, img2_batch):
    """
    Compute the Mean Squared Error (MSE) map between one image and a batch of images. 
    The MSE map provides a spatial representation of the squared differences.

    Parameters:
    - img1 (torch.Tensor): First input image as a torch tensor with shape (C, H, W) or Batch of first input images as a torch tensor with shape (B, C, H, W).
    - img2_batch (torch.Tensor): Batch of second input images as a torch tensor with shape (B, C, H, W).

    Returns:
    - torch.Tensor: A batch of MSE maps with shape (B, H, W), representing the MSE at each spatial location.
    """

    # Add an extra batch dimension to img1 to match the (B, C, H, W) format
    if img1_batch.dim() == 3:
        img1_batch = img1_batch.unsqueeze(0)
    if img2_batch.dim() == 3:
        img2_batch = img2_batch.unsqueeze(0)


    # Compute the squared differences
    squared_diff = (img1_batch - img2_batch) ** 2

    mse = average_nonzero(squared_diff, dims=[1, 2, 3])

    # Compute the MSE map by averaging the squared differences across the channel dimension
    return mse


def compute_lpips_similarity_index(img1_batch, img2_batch, net_type='alex', batch_size=40):
    """
    Compute the LPIPS similarity between two batch of images.

    Parameters:
    - img1_batch (torch.Tensor): Batch of first input images as a torch tensor with shape (B, C, H, W) or (1, C, H, W).
    - img2_batch (torch.Tensor): Batch of second input images as a torch tensor with shape (B, C, H, W).
    - net_type (str): Type of network to use for LPIPS computation ('alex', 'vgg', etc.).

    Returns:
    - torch.Tensor: LPIPS similarity values for each image in the batch.
    """
    # Initialize the LPIPS model with the specified network type
    lpips_model = lpips.LPIPS(net=net_type)

    if img1_batch.dim() == 3:
        img1_batch = img1_batch.unsqueeze(0)
    if img2_batch.dim() == 3:
        img2_batch = img2_batch.unsqueeze(0)

    # Move the LPIPS model to the same device as the input tensors

    # if the bacth size more that 40 we need to split the batch to smaller batches
    if (img1_batch.shape[0] > batch_size) and (img2_batch.shape[0] > batch_size):
        img1_batch_split = torch.split(img1_batch, batch_size)
        img2_batch_split = torch.split(img2_batch, batch_size)
        similarity_lpips = list()
        for i in range(len(img1_batch_split)):
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                lpips_model = lpips_model.to('cuda')
                img1_batch_small = img1_batch_split[i].to('cuda')
                img2_batch_small = img2_batch_split[i].to('cuda')
            else:
                img1_batch_small = img1_batch_split[i]
                img2_batch_small = img2_batch_split[i]
            similarity_lpips.append(lpips_model(img1_batch_small, img2_batch_small))
        similarity_lpips = torch.cat(similarity_lpips)
    elif (img2_batch.shape[0] > batch_size) and (img1_batch.shape[0] <= batch_size):
        img2_batch_split = torch.split(img2_batch, batch_size)
        similarity_lpips = list()
        for i in range(len(img2_batch_split)):
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                lpips_model = lpips_model.to('cuda')
                img2_batch_small = img2_batch_split[i].to('cuda')
                img1_batch = img1_batch.to('cuda')
            else:
                img2_batch_small = img2_batch_split[i]
            similarity_lpips.append(lpips_model(img1_batch, img2_batch_small))
        similarity_lpips = torch.cat(similarity_lpips)
    else:
        if torch.cuda.is_available():
            lpips_model = lpips_model.to('cuda')
            img1_batch = img1_batch.to('cuda')
            img2_batch = img2_batch.to('cuda')
        similarity_lpips = lpips_model(img1_batch, img2_batch)


    # Compute LPIPS similarity
    return similarity_lpips