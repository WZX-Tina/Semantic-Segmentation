import numpy as np
import os

parent_folder = '../train_data/train'

for dirs in os.listdir(parent_folder):
    print(dirs)
    for filename in os.listdir(os.path.join(parent_folder,dirs)):
        if filename.endswith('.npy'):
            mask_path = os.path.join(parent_folder,dirs,filename)
            mask = np.load(mask_path)
            print(mask.shape)
            print(mask)
            print(np.sum(mask))
            print(np.max(mask))

