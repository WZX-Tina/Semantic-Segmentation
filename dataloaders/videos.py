from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import torch
import cv2
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms


class VideosDataset(BaseDataSet):
    """
    Videos dataset 
    nyu.edu
    """
    def __init__(self, **kwargs):
        self.num_classes = 48
        self.palette = palette.Videos_palette
        super(VideosDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in  ["train", "val"]:
            # print(os.getcwd())
            self.image_dir = os.path.join(self.root, self.split)
            self.files = []
            self.dirs = []
            for dirs in os.listdir(self.image_dir):
                # print(dirs)
                if dirs != '.DS_Store':
                    for filename in os.listdir(os.path.join(self.image_dir, dirs)):
                        if filename.endswith(('.png')):
                            self.files.append(filename.split('.')[0][6:])
                            self.dirs.append(dirs)
            # print(self.dirs)
            # print(self.frames)
        else: raise ValueError(f"Invalid split name {self.split}")
        print('_set_files complete')

    def _load_data(self, index):
        print('_load_data begins')
        image_id = os.path.join(self.dirs[index], 'image_' + self.files[index])
        image_path = os.path.join(self.image_dir, image_id + '.png')
        label_path = os.path.join(self.image_dir, self.dirs[index], 'mask.npy')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.load(label_path)[int(self.files[index])] - 1 # from -1 to 47
        print('image shape ', image.shape)
        print('label shape ', label.shape)
        print(image_id)
        return image, label, image_id

class Videos(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.48897059, 0.46548275, 0.4294]
        self.STD = [0.22861765, 0.22948039, 0.24054667]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = VideosDataset(**kwargs)
        print(len(self.dataset))
        print(self.dataset)
        super(Videos, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
