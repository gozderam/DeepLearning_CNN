import os
import py7zr
import data_utils.data_preparation as data_prep
from data_utils.cifar10_dataset import CIFAR10Dataset
from data_utils.cifar10_testset import CIFAR10Testset
import torch
import torchvision


def load_train_data(transform, batch_size, data_dir='../data'):
    
    images_root_dirs = [f'{data_dir}/train', f'{data_dir}/train_aug']
    labels_csv_files = [f'{data_dir}/trainLabels.csv', f'{data_dir}/trainLabels_aug.csv']
    
    if not os.path.isdir(images_root_dirs[0]):
        data_prep.prepare_data(data_dir)

    trainset = CIFAR10Dataset(labels_csv_files=labels_csv_files, images_root_dirs=images_root_dirs, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    valset = CIFAR10Dataset(labels_csv_files=[f'{data_dir}/valLabels.csv'], images_root_dirs=[f'{data_dir}/validation'], transform=transform)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, valloader


def load_torch_train_data(transform, batch_size, data_dir='../data_torch'):

    print ('loading started')
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                                train=True, 
                                                transform=transform,
                                                download=True)
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=batch_size, 
                                        shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size= batch_size, 
                                            shuffle=False)
    print ('loading finished')
    return train_loader, val_loader


def load_test_data(transform, batch_size, data_dir='../data'):
    
    images_root_dir = f'{data_dir}/test'
    
    if not os.path.isdir(images_root_dir):
        with py7zr.SevenZipFile(f'{data_dir}/test.7z', mode='r') as z:
            z.extractall(path=data_dir)

    testset = CIFAR10Testset(images_root_dir=images_root_dir, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return testloader, testset.number_of_images

