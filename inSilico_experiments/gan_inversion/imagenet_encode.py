
from argparse import ArgumentParser
import warnings
import os
import sys
import numpy as np
import torch
import time
import shutil
import pickle

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

    from core.utils.image_similarity import *
    from core.utils.CNN_scorers import TorchScorer

    def get_activations(path, format):
    
        ref_img_nms, ref_img_tsr = load_ref_imgs(
        imgdir=path, preprocess_type='center_crop', image_size=256, Nlimit=1000)
        
        #images = [im_name for im_name in os.listdir(path) if im_name[-len(format):]==format]                       
        #batch  = torch.from_numpy(np.asarray([cv2.imread(os.path.join(path, image)).transpose(2,0,1) for image in images]))

        sim_scorer = TorchImageDistance()
        sim_scorer.first_image_batch = ref_img_tsr

        net_layer_dict = sim_scorer.get_CCN_encoding()
        
        return net_layer_dict, ref_img_nms

    # let parse the arguments
    args = parser.parse_args() 
    gan_name = args.gan_name
    task_id = args.task_id
    folder_name = folders[task_id]
    max_iter = int(args.max_iter)

    samples = [f"ILSVRC2012_test_{i:08}.JPEG" for i in np.random.choice(np.arange(1,100001), size=10000, replace=False)]

    batch_size = 500

    net_layer_dict_list = [] 
    img_nms_list = []

    for i in range(int(10000/500)):
         
        group_name = f"Group_{str(i)}"

        group_samples = samples[i*500:i*500+500]

        data_path = os.path.join(data_root, group_name)
        if not os.path.isdir(data_path):
            os.makedirs(data_path)

        for sample in group_samples:
            shutil.copyfile(os.path.join(full_data_root, sample), os.path.join(data_path, sample))
   
        net_layer_dict, ref_img_nms = get_activations(data_path, '.JPEG')

        net_layer_dict_list.append(net_layer_dict)
        img_nms_list.extend(ref_img_nms)

    total_imagenet_dict = {}
    first_dict = net_layer_dict_list[0]

    for net, layers in first_dict.items():

        layer_dict = {}
        for layer in layers.keys():

            all_groups = np.vstack(tuple([net_layer[net][layer] for net_layer in net_layer_dict_list]))

            layer_dict[layer] = all_groups

        total_imagenet_dict[net] = layer_dict

    with open("/n/home09/dsprague/data/net_layer_activations/imagenet_full", "wb") as handle:
        pickle.dump(total_imagenet_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    

    

    