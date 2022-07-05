import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List
from PIL import Image
from skimage import io

class CIFAR10Dataset(Dataset):

    class_to_label_number = {
            'airplane'  : 0,
            'automobile': 1,
            'bird'      : 2,
            'cat'       : 3,
            'deer'      : 4,
            'dog'       : 5,
            'frog'      : 6,
            'horse'     : 7,
            'ship'      : 8,
            'truck'     : 9
        }

    def __init__(self, labels_csv_files: List[str], images_root_dirs: List[str], transform=None, start_img_name=1):
        self.labels_frames = [pd.read_csv(labels_csv_file, index_col='id') for labels_csv_file in labels_csv_files]
        self.images_root_dirs = images_root_dirs
        self.parts_cum_counts = np.cumsum([labels_frame.shape[0] for labels_frame in self.labels_frames])
        self.transform = transform
        self.start_img_name = start_img_name

    def __len__(self):
        return self.parts_cum_counts[-1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        part_idx = next(ci for ci, cval in enumerate(self.parts_cum_counts) if idx + 1 <= cval)
        images_root_dir = self.images_root_dirs[part_idx]
        labels_frame = self.labels_frames[part_idx]

        img_name = os.path.join(images_root_dir, str((self.start_img_name - 1) + (idx + 1)) + '.png')
        image = Image.fromarray(io.imread(img_name))

        label = CIFAR10Dataset.class_to_label_number[labels_frame.loc[idx + 1][0]]

        if self.transform:
            sample = [self.transform(image), label]
        else:
            sample = [image, label]
        
        return sample