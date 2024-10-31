
from argparse import ArgumentParser
import warnings
import os
import sys
import numpy as np
import torch
import time
import shutil

parser = ArgumentParser()
parser.add_argument("--gan_name", type=str, default="fc6", help="GAN model name")
#parser.add_argument("--folder_name", type=str, default="monkey_grooming", help="name of the folder with the image batch files")
parser.add_argument("--max_iter", type=int, default=int(5E5), help="Number gradient descent iterations")
parser.add_argument("--task_id", type=int, default=0, help="task id number")
#parser.add_argument("--folders", type='str', default=[], help='folders to iterate through' )

save_data_main_root = r"/n/home09/dsprague/data/imagenet_test_inverted"
full_data_root = r"/n/holylfs06/LABS/kempner_shared/Lab/data/imagenet_1k/ILSVRC2012_img_test_v10102019/test"
data_root = r"/n/home09/dsprague/data/imagenet_test"

sys.path.append(r"/n/home09/dsprague/Ponce_rotation")

os.makedirs(save_data_main_root, exist_ok=True)
os.makedirs(data_root, exist_ok=True)

folders = ['Group1', 'Group2', 'Group3', 'Group4']

if __name__=="__main__":
    # let get log of the time
    t_save = time.time()
    # importing tje required librari
    from core.utils.func_lib import *
    from core.utils.GAN_utils import upconvGAN
    from core.utils.GAN_utils import loadBigGAN, BigGAN_wrapper
    from core.utils.GAN_invert_utils import *
    from core.utils.GAN_utils import upconvGAN


    # let parse the arguments
    args = parser.parse_args() 
    gan_name = args.gan_name
    task_id = args.task_id
    folder_name = folders[task_id]
    max_iter = int(args.max_iter)

    # load the GAN model
    if gan_name == 'BigGAN':
        BGAN = loadBigGAN()
        G = BigGAN_wrapper(BGAN)
        code_length = 256
        fixnoise = 0.7 * truncated_noise_sample(1, 128, seed=1)
        init_code = np.concatenate((fixnoise, np.zeros((1, 128))), axis=1)
        init_code = torch.tensor(init_code).float().cuda()
    elif gan_name == 'fc6':
         G = upconvGAN('fc6').cuda().eval()
    else:
        raise ValueError("The GAN model name is not recognized")
    

    data_path = os.path.join(data_root, folder_name)
    if not os.path.isdir(data_path):
         os.makedirs(data_path) 

         samples = [f"ILSVRC2012_test_{i:08}.JPEG" for i in np.random.choice(np.arange(25000*task_id+1,25000*task_id+25000+1), size=250, replace=False)]

    for sample in samples:
         shutil.copyfile(os.path.join(full_data_root, sample), data_path)
            
    
    # load the image batch
    img_batch_path = data_path
    ref_img_nms, ref_img_tsr = load_ref_imgs(
        imgdir=img_batch_path, preprocess_type='center_crop', image_size=256)
    
    # invert the image batch
    if gan_name == 'fc6':
            z_opts, img_opts= GAN_invert(G, ref_img_tsr.cuda(), max_iter=int(max_iter),
                                    print_progress=False, batch_size = ref_img_tsr.shape[0])
    elif gan_name == 'BigGAN':

            z_opts, img_opts = GAN_invert(G, ref_img_tsr.cuda(), z_init=init_code, max_iter=int(max_iter),
                                    print_progress=False, batch_size = ref_img_tsr.shape[0])
    else:
        raise ValueError("The GAN model name is not recognized")
    
    # let save the inverted images and the codes and the reference images
    save_dir = save_data_main_root
    os.makedirs(save_dir, exist_ok=True)
    img_format = "png"
    for imgid, imgs in enumerate(ref_img_tsr):
        ref_img = ToPILImage()(imgs)
        img_name = ref_img_nms[imgid].partition(".")[0]
        ref_img.save(join(save_dir, f"{img_name}_original.{img_format}"))
        # let save the inverted image
        inv_img = ToPILImage()(img_opts[imgid])
        inv_img.save(join(save_dir, f"{img_name}_inverted.{img_format}"))
        # let save the inverted code
        z_opt = z_opts[imgid]
        z_opt = z_opt.cpu().detach().numpy()
        np.save(join(save_dir, f"{img_name}_code.npy"), z_opt)
    t_end = time.time()
    t_took_sec = t_end - t_save
    t_took_min = t_took_sec / 60
    print(f"Video {folder_name} is processed and saved. everything took {t_took_min} minutes")


    

    

    