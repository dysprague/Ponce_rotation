import os
import sys
sys.path.append(r"/n/home09/dsprague/Ponce_rotation")
from core.utils.func_lib import *
from core.utils.GAN_utils import upconvGAN
from core.utils.GAN_invert_utils import GAN_invert
from core.utils.GAN_utils import upconvGAN
import numpy as np

from PIL import Image
import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize, ToPILImage, CenterCrop

def GAN_inf_movies(G, z_init, z_frames, output_dir):
    # Find the distance between Z[i-1] and Z[i] and then move in a random direction away by that same distance

    z_init = torch.from_numpy(z_init)

    z_opt = z_init.clone().detach().requires_grad_(False).to("cuda")
    
    for i in range(len(z_frames)):

        if i == 0:    
            z_use = z_opt
            img_opt = G.visualize(z_use)

        else: 
            enc_dist = np.linalg.norm(z_frames[i] - z_frames[i-1])

            random_numbers = np.random.uniform(-1, 1, z_init.shape[0])
            rand_update = (random_numbers/np.linalg.norm(random_numbers))*enc_dist

            z_use = z_opt + torch.from_numpy(rand_update).to("cuda")

            img_opt = G.visualize(z_use)

            z_opt = z_use

        file_name = f"frame_{i}_inv"
        image_format = "png"
        img = ToPILImage()(img_opt[0])
        img.save(os.path.join(output_dir, f"{file_name}.{image_format}"))
        
        z_save= z_opt.clone()
        z_save = z_save.cpu().detach().numpy()

        np.save(os.path.join(output_dir, f"{file_name}.npy"), z_save)

if __name__ == '__main__':

    G = upconvGAN("fc6").cuda().eval()

    data_path = r"/n/home09/dsprague/data/videos_inverted"

    for folder in os.listdir(data_path):
        if not os.path.isdir(os.path.join(data_path, folder)):
            continue 

        z_frames = [np.load(os.path.join(data_path, folder, file)) for file in os.listdir(os.path.join(data_path, folder)) if file[-4:]=='.npy']

        z_init = z_frames[0]

        output_dir = os.path.join(f"/n/home09/dsprague/data/videos_random_gen/{folder}")
        os.makedirs(output_dir, exist_ok=True)

        GAN_inf_movies(G, z_init, z_frames, output_dir)



