import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tqdm
import py7zr
import os
from sklearn.model_selection import train_test_split
from data_utils.cifar10_dataset import CIFAR10Dataset

# https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    def __init__(self, n_holes, length, p):
        self.n_holes = n_holes
        self.length = length
        self.p = p

    def __call__(self, img):
        if np.random.binomial(1, self.p) == 0:
            return img

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def _offline_augmentate(src_labels_csv_file, src_images_root_dir, dest_labels_csv_file, dest_images_root_dir, transformations):
    os.makedirs(dest_images_root_dir)
    
    transformed_cifar10ds = CIFAR10Dataset(labels_csv_files=[src_labels_csv_file], images_root_dirs=[src_images_root_dir], transform=transformations)
    batch_size = 5
    dataloader = DataLoader(transformed_cifar10ds, batch_size=batch_size, shuffle=False, num_workers=0)

    dest_labels = [None] * len(transformed_cifar10ds)
    dest_img_names = [None] * len(transformed_cifar10ds)

    for i, data in enumerate(tqdm.tqdm(dataloader)):    
        for idx in range(batch_size):
            img = data[0][idx]
            label = CIFAR10Dataset.label_number_to_class[data[1][idx].item()]
            img_name = str(len(transformed_cifar10ds) + batch_size * i + idx + 1)

            dest_labels[batch_size * i + idx] = label
            dest_img_names[batch_size * i + idx] = img_name
            img = img / 2 + 0.5 # denormalize
            transform = transforms.ToPILImage()
            transform(img).save(os.path.join(dest_images_root_dir, img_name + '.png'))

    labels_frame = pd.DataFrame({'id': dest_img_names, 'label': dest_labels})
    labels_frame.to_csv(dest_labels_csv_file, index=False)


def _create_validation_dataset(src_labels_csv_file, src_images_root_dir, dest_labels_csv_file, dest_images_root_dir):
    labels = pd.read_csv(src_labels_csv_file, index_col='id')
    train_labels, val_labels = train_test_split(labels, test_size=5000, random_state=42)
    train_labels.reset_index(inplace=True)
    val_labels.reset_index(inplace=True)  
    
    os.makedirs(dest_images_root_dir)
    for index, img in val_labels.iterrows():
        os.rename(f'{src_images_root_dir}/{img.id}.png', f'{dest_images_root_dir}/{index + 1}.png')
    os.makedirs(f'{src_images_root_dir}2')
    for index, img in train_labels.iterrows():
        os.rename(f'{src_images_root_dir}/{img.id}.png', f'{src_images_root_dir}2/{index + 1}.png')
    for index, img in train_labels.iterrows():
        os.rename(f'{src_images_root_dir}2/{index + 1}.png', f'{src_images_root_dir}/{index + 1}.png')
    os.rmdir(f'{src_images_root_dir}2')
    
    train_labels = pd.DataFrame({'id': train_labels.index + 1, 'label': train_labels.label})
    val_labels = pd.DataFrame({'id': val_labels.index + 1, 'label': val_labels.label})
    train_labels.to_csv(src_labels_csv_file, index=False)
    val_labels.to_csv(dest_labels_csv_file, index=False)


def prepare_data(directory):
    with py7zr.SevenZipFile(f'{directory}/train.7z', mode='r') as z:
        z.extractall(path=directory)
    _create_validation_dataset(f'{directory}/trainLabels.csv', f'{directory}/train', f'{directory}/valLabels.csv', f'{directory}/validation')
    
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(p=0.9),
        transforms.RandomApply([transforms.ColorJitter()], p=0.4),
        transforms.RandomApply([transforms.RandomRotation(degrees=(-20, 20))], p=0.4),
        Cutout(2, 10, p=0.4)
    ])

    _offline_augmentate(f'{directory}/trainLabels.csv', f'{directory}/train/', f'{directory}/trainLabels_aug.csv', f'{directory}/train_aug/', transformations)
    