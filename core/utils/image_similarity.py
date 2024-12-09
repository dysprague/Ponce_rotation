
from core.utils.CNN_scorers import TorchScorer
from sklearn.metrics.pairwise import paired_cosine_distances
from core.utils.func_lib import *
from core.utils.basic_functions import *

import numpy as np
import torch
import lpips

class TorchImageDistance:
    # this class calculates the PAIRED similarity between two images with different merics
    # The out put is the mean similarity the distance between the two images

    def __init__(self):
        self.first_image_batch = None # the first image tensor batch
        self.second_image_batch = None # the second image tensor batch
        self.curent_slice = None 
        self.torch_scorer_list = []

    def set_first_image_batch(self, first_image_batch):
        # if the image is not B, C, H, W, then we need to unsqueeze it
        if len(first_image_batch.shape) == 3:
            first_image_batch = np.expand_dims(first_image_batch, axis=0)
        self.first_image_batch = first_image_batch

    def set_second_image_batch(self, second_image_batch):
        # if the image is not B, C, H, W, then we need to unsqueeze it
        if len(second_image_batch.shape) == 3:
            second_image_batch = np.expand_dims(second_image_batch, axis=0)
        self.second_image_batch = second_image_batch
    def __batch_size_check(self):
        if not(self.first_image_batch.shape[0] == self.second_image_batch.shape[0]):
            raise ValueError('The batch size of the two images should be the same')
    def get_images_batch(self):
        return self.first_image_batch, self.second_image_batch
    
    def __set_CNN_scorer_list(self, net_name_list=None, layers_list=None):
         # define the CNN models and layers to be used
        if net_name_list is None:
            self.net_name_list = ['resnet50', 'resnet50_linf_8', 'cornet_s', 'alexnet']
        if layers_list is None:
            # THE NUMBER OF LAYERS SHOULD BE THE SAME FOR ALL NETWORKS
            self.layers_list = [['.layer1.2.BatchNorm2dbn3', '.layer2.3.BatchNorm2dbn3', '.layer3.5.BatchNorm2dbn3', '.layer4.2.BatchNorm2dbn3'],
                            ['.layer1.2.BatchNorm2dbn3', '.layer2.3.BatchNorm2dbn3', '.layer3.5.BatchNorm2dbn3', '.layer4.2.BatchNorm2dbn3'],
                            ['.V1.BatchNorm2dnorm2', '.V2.BatchNorm2dnorm3_1', '.V4.BatchNorm2dnorm3_3', '.IT.BatchNorm2dnorm3_1'],
                            ['.features.Conv2d3', '.features.Conv2d6', '.features.Conv2d8', '.features.Conv2d10']]
        
        for net_name, layers in zip(self.net_name_list, self.layers_list):
            # load the CNN model
            scorer = TorchScorer(net_name)
            self.torch_scorer_list.append(scorer)
        print('The CNN scorers are set')

    def __set_encoding_slice(self, units_slice): 
        # clean the memory if the slice is not None and changed
        if (self.curent_slice is not None) and (not(self.curent_slice == units_slice)):
            print(f'The units_slice is changed to {self.curent_slice} so we need to cleanup the memory')
            for scorer in self.torch_scorer_list:
                scorer.cleanup()
            self.__set_CNN_scorer_list()

        self.curent_slice = units_slice
        if self.curent_slice == 'center':
            for scorer, layer_list in zip(self.torch_scorer_list, self.layers_list):
                _, _ = set_all_center_unit_population_recording(
                                                scorer, layer_list, print_info=False)
        elif self.curent_slice == 'all':
            for scorer, layer_list in zip(self.torch_scorer_list, self.layers_list):
                _, _ = set_all_unit_population_recording(
                                                scorer, layer_list, print_info=False)
        else:
            raise ValueError('The units_slice is not supported')

    def get_CCN_encoding(self, units_slice = 'center'):
        
        if len(self.torch_scorer_list) == 0:
            self.__set_CNN_scorer_list()

        if not (self.current_slice == units_slice):
            self.__set_encoding_slice(units_slice)

        encoded_image_batch_list = []
        for scorer, layer_list in zip(self.torch_scorer_list, self.layers_list):
            encoded_image_batch, _ = encode_image(scorer, self.first_image_batch, key=layer_list, RFresize=False, cat_layes=False)
            encoded_image_batch_list.append(encoded_image_batch)

        net_layer_dict = {}

        for i in range(len(self.net_name_list)):
            net_layer_dict[self.net_name_list[i]] = {}
            for j in range(len(self.layers_list[0])):
                net_layer_dict[self.net_name_list[i]][self.layers_list[i][j]] = encoded_image_batch[i][j]

        return net_layer_dict

    def get_CCN_distance(self, similarity='cosine', units_slice='center'):
        # this function is used to calculate the similarity between two images batch using the CNN features
        # we consider 4 (or other same number of) convilutional layers from each network to calculate the similarity lock at the set_CNN_scorer_list

        # let check if the batch size is the same
        self.__batch_size_check()

        # let set the score list if it is not set
        if len(self.torch_scorer_list) == 0:
            self.__set_CNN_scorer_list()
        
        if not (self.curent_slice == units_slice):
            self.__set_encoding_slice(units_slice)
        
        # let encode the images
        encoded_first_image_batch_list = []
        encoded_second_image_batch_list = []
        for scorer, layer_list in zip(self.torch_scorer_list, self.layers_list):
            encoded_first_image_batch, _ = encode_image(scorer, self.first_image_batch, key=layer_list,
                                            RFresize=False, cat_layes=False)
            encoded_second_image_batch, _ = encode_image(scorer, self.second_image_batch, key=layer_list,
                                            RFresize=False, cat_layes=False)
            encoded_first_image_batch_list.append(encoded_first_image_batch)
            encoded_second_image_batch_list.append(encoded_second_image_batch)

        #return encoded_first_image_batch_list, encoded_second_image_batch_list
        
        # let calculate the similarity
        similarity_matrix = np.zeros((len(self.net_name_list), len(self.layers_list[0]), self.first_image_batch.shape[0]))
        #print(f'the shape of the similarity matrix is {similarity_matrix.shape}')
        for i in range(len(self.net_name_list)):
            for j in range(len(self.layers_list[0])):
                if similarity == 'cosine':
                    #print(f'Calculating the cosine similarity for {self.net_name_list[i]}, {self.net_name_list[j]}')
                    similarity_matrix[i, j] = paired_cosine_distances(encoded_first_image_batch_list[i][j], encoded_second_image_batch_list[i][j])
                else:
                    raise ValueError('The similarity is not supported')
        
        return np.mean(similarity_matrix, axis=(0,1)), similarity_matrix
    
    #
    def get_L2_distance(self):

        # let check if the batch size is the same
        self.__batch_size_check()
        squared_diff = (self.first_image_batch - self.second_image_batch)**2
        l2_distance = torch.sqrt(torch.sum(squared_diff, dim=(1,2,3)))

        return l2_distance, squared_diff
    
    def get_MSE_distance(self):

        # let check if the batch size is the same
        self.__batch_size_check()
        squared_diff = (self.first_image_batch - self.second_image_batch)**2
        mse_distance = average_nonzero(squared_diff, dims=[1, 2, 3])

        return mse_distance, squared_diff
    def __compute_lpips_similarity(self, lpips_model, img1_batch, img2_batch):
        if torch.cuda.is_available():
            lpips_model = lpips_model.to('cuda')
            img1_batch = img1_batch.to('cuda')
            img2_batch = img2_batch.to('cuda')
        similarity_lpips = lpips_model(img1_batch, img2_batch)
        return similarity_lpips

    def get_LPIPS_distance(self, net_type='alex', max_batch_size=40):

        # let check if the batch size is the same
        self.__batch_size_check()

        lpips_model = lpips.LPIPS(net=net_type)

        # if the image bach is more than max_batch_size we need to split it to smaller batches
        if self.first_image_batch.shape[0] > max_batch_size:
            first_image_batch_split = torch.split(self.first_image_batch, max_batch_size)
            second_image_batch_split = torch.split(self.second_image_batch, max_batch_size)
            similarity_lpips = list()
            for i in range(len(first_image_batch_split)):
                similarity_lpips.append(self.__compute_lpips_similarity(lpips_model, first_image_batch_split[i], second_image_batch_split[i]))
            similarity_lpips = torch.cat(similarity_lpips)
        else:
            similarity_lpips = self.__compute_lpips_similarity(lpips_model, self.first_image_batch, self.second_image_batch)
        
        return similarity_lpips.squeeze().cpu().detach()
    

    def memory_cleanup(self):
        if not(len(self.torch_scorer_list) == 0):
            for scorer in self.torch_scorer_list:
                scorer.cleanup()
        torch.cuda.empty_cache() 
             
