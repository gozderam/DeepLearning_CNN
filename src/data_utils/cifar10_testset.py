import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage import io

class CIFAR10Testset(Dataset):

    label_number_to_class = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }

    def __init__(self, images_root_dir: str, transform=None, start_img_name=1):
        self.images_root_dir = images_root_dir
        self.number_of_images = len([name for name in os.listdir(self.images_root_dir) if os.path.isfile(os.path.join(self.images_root_dir, name))])
        self.transform = transform
        self.start_img_name = start_img_name

    def __len__(self):
        return self.number_of_images

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_root_dir, str((self.start_img_name - 1) + (idx + 1)) + '.png')
        image = Image.fromarray(io.imread(img_name))

        if self.transform:
            sample = [self.transform(image)]
        else:
            sample = [image]
        
        return sample