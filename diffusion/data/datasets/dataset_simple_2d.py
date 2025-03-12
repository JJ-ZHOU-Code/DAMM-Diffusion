
import os
import torch.utils.data as data 
import torch 
from torch import nn
from pathlib import Path 
from torchvision import transforms as T
import pandas as pd 

from PIL import Image
import torchvision.transforms.functional as TF
from random import random

class Dataset_Paired(data.Dataset):
    def __init__(
        self,
        path_root,
        transform = None,
        image_resize = None,
    ):
        super().__init__()
        self.path_root = path_root
        if not os.path.exists(self.path_root):
            print(self.path_root)
            raise Exception(f"[!] dataset is not exited")

        self.image_file_name = sorted(os.listdir(os.path.join(self.path_root, 'QD')))
        
        if transform is None: 
            self.transform = T.Compose([
                T.Resize(image_resize) if image_resize is not None else nn.Identity(),
                T.ToTensor(),
                T.Normalize(mean=0.5, std=0.5) # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_file_name)

    def __getitem__(self, index):
        
        file_name = self.image_file_name[index]
        img_QD = Image.open(os.path.join(self.path_root, 'QD', file_name)).convert('RGB')
        img_CD31 = Image.open(os.path.join(self.path_root, 'CD31', file_name)).convert('RGB')
        img_DAPI = Image.open(os.path.join(self.path_root, 'DAPI', file_name)).convert('RGB')

        return {'source':self.transform(img_QD), 'QD': self.transform(img_QD), 'CD31': self.transform(img_CD31), 'DAPI': self.transform(img_DAPI), 'file_name':file_name}
    