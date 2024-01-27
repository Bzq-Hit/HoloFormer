
import torch
import torch.utils.data as data
import os
import random
import numpy as np
from utils import Augment_RGB_torch
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

def get_training_data(data_dir): 
    return DatasetFromFolder_realnimag_inverse_withoutnorm(data_dir, Train=True)
        

def get_validation_data(data_dir):
    return DatasetFromFolder_realnimag_inverse_withoutnorm(data_dir, Train=False)


class DatasetFromFolder_realnimag_inverse_withoutnorm(data.Dataset):
    def __init__(self, data_dir, Train = True):
        super(DatasetFromFolder_realnimag_inverse_withoutnorm, self).__init__()
        self.data_dir = data_dir
        self.Train = Train

        self.data_all_dir = os.listdir(data_dir)
        self.numpy_list = []
        for i in self.data_all_dir:
            if i.endswith('.npy'):
                self.numpy_list.append(i) 
        
        self.ob_real_list = []
        for i in self.numpy_list:
            if 'ob_real' in i:
                self.ob_real_list.append(i) 
        
        random.seed(2021)
        random.shuffle(self.ob_real_list)

        self.train_ob_real_list = self.ob_real_list[0:int(len(self.ob_real_list)*0.9)] 
        self.val_ob_real_list = self.ob_real_list[int(len(self.ob_real_list)*0.9):]


        self.gt_dic = {}
        for i in self.ob_real_list:
            self.gt_dic[i] = (i.replace('ob_real', 'GT_real'), i.replace('ob_real', 'GT_imag')) 

        self.ob_imag_dict = {}
        for i in self.ob_real_list:
            self.ob_imag_dict[i] = i.replace('ob_real', 'ob_imag') 
    
    def __getitem__(self, index):
        if self.Train:
            ob_real_path = self.train_ob_real_list[index]
            ob_imag_path = self.ob_imag_dict[ob_real_path]
            gt_real_path = self.gt_dic[ob_real_path][0]
            gt_imag_path = self.gt_dic[ob_real_path][1]
        else:
            ob_real_path = self.val_ob_real_list[index]
            ob_imag_path = self.ob_imag_dict[ob_real_path]
            gt_real_path = self.gt_dic[ob_real_path][0]
            gt_imag_path = self.gt_dic[ob_real_path][1]
        
        ob_real = torch.from_numpy(np.load(os.path.join(self.data_dir, ob_real_path)))
        ob_imag = torch.from_numpy(np.load(os.path.join(self.data_dir, ob_imag_path)))
        gt_real = torch.from_numpy(np.load(os.path.join(self.data_dir, gt_real_path)))
        gt_imag = torch.from_numpy(np.load(os.path.join(self.data_dir, gt_imag_path)))

        if self.Train:
            apply_trans = transforms_aug[random.getrandbits(3)]
            ob_real = getattr(augment, apply_trans)(ob_real)
            ob_imag = getattr(augment, apply_trans)(ob_imag)
            gt_real = getattr(augment, apply_trans)(gt_real)
            gt_imag = getattr(augment, apply_trans)(gt_imag)
            return ob_real, ob_imag, gt_real, gt_imag
        else:
            return ob_real, ob_imag, gt_real, gt_imag
        
    def __len__(self):
        if self.Train:
            return len(self.train_ob_real_list)
        else:
            return len(self.val_ob_real_list)






