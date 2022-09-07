"""
Details
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
import numpy as np
import random
import os
import csv
from PIL import Image

import torchnet as tnt
import torch
import torch.utils.data as data
import torchvision.transforms as T
from torch.utils.data.dataloader import default_collate

from collators

# =============================================================================================== #
# Classes
# =============================================================================================== #
# ------ Data set --------------------------------------- #
# ======================================================= #
class DevelopmentSet(data.Dataset):
    """
    Detials
    """
    def __init__(self, root, split, transforms=None, target_transform=None):
        """
        Detials
        """
        self.root = os.path.expanduser(root)
        self.data_folder = os.path.join(self.root, 'data', 'vision')
        self.split_folder = os.path.join(self.root, 'name')
        assert(split == 'train' or split == 'val')
        split_csv_file = os.path.join(self.split_folder, split + 'csv_file.csv')

        self.transforms = transforms
        self.target_transfroms = target_transform
        with open(split_csv_file, 'r') as f:
            reader = csv.read(f, delimiter = "delim_goes_here")
            self.img_file = []
            self.labels = []
            for row in reader:
                self.img_file.append(row[0])
                self.labels.append(int(row[1]))


    def __getitem__(self, index):
        """
        Details
        """
        image_path = os.path.join(self.data_folder, self.img_file[index])   
        img = Image.open(image_path).convert('RGB')
        target = self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transfroms is not None:
            target = self.target_transfroms(target)
        
        return img, target
        

    def __len__(self):
        return len(self.labels)


class DatasetManager(data.Dataset):
    """
    Detials
    """
    def __init__(self, dataset_name, split, random_size_crop=False):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_size_crop = random_size_crop

        # Dataset names will go here. for now just the devel dataset
        if self.dataset_name == "DevelopmentDataset":
            if self.split != 'train':
                transform_list_augmentation = [T.CentreCrop(224)]
            else:
                if self.random_size_crop:
                    transform_list_augmentation = [T.RamdomResizeCrop(224),
                                                   T.RandomHorizontalFlip()]
                else:
                    transform_list_augmentation = [T.RandomCrop(224),
                                                   T.RandomHorizontalFlap()]

            # ImageNet mean and var for ImageNet pretrained models.
            self.mean_pixel = [0.485, 0.456, 0.406]
            self.std_pixel = [0.229, 0.224, 0.225]
            transform_list_normalized = [T.ToTensor(),
                                           T.Normalize(mean = self.mean_pixel,
                                                       std = self.std_pixel)]

            self.transform_augmentation_normalize = T.Compose(transform_list_augmentation + 
                                                              transform_list_normalized)
            self.data = DevelopmentSet(root = "put_root_here", 
                                       split = self.split, 
                                       transforms = self.transform_augmentation_normalize)
        else:
            raise ValueError("Not recognised dataset {0}".format(self.dataset_name))


    def __getitem__(self, index):
        img, label = self.data[index]
        return img, int(label), index 


    def __len__(self):
        return len(self.data)                                         


# ------ Data loader ------------------------------------ #
# ======================================================= #
class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Dataloader():
    """
    Details 
    """
    def __init__(self,
                 dataset,
                 batch_size=1,
                 unsupervised=True,
                 num_workers=0,
                 shuffle=True):
        """
        Detials
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.unsupervised = unsupervised
        self.epoch_size = len(dataset)
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.random_seed = 42
        self.data_loader_iter = self.get_iter()

        self.inv_transform = T.Compose([
            Denormalize(self.dataset.mean_pix, self.dataset.std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])


    def get_iter(self):
        """
        Details
        """
        # setting random seed
        random.seed(self.random_seed)

        if self.unsupervised:
            # loader function for unsupervised mode, the loader returns:
            # - image and index to image in dataset
            # - label of rotation:
            # - - 0 = 0 deg
            # - - 1 = 90 deg
            # - - 2 = 180 deg 
            # - - 3 = 270 deg
            # 4 coppies should be created durin the models forward pass
            def _load_funtion(idx):
                idx = idx % len(self.dataset)
                img, _, index = self.dataset[idx]
                rotation_labels = torch.LongTensor([0, 1, 2, 3])
                image_includes = torch.LongTensor([index, index, index, index])
                return img, rotation_labels, image_includes

            
            def _collate_fun(batch):
                batch = default_collate(batch)
                assert(len(batch)==3)
                batch_size, rotation = batch[1].size()
                batch[1] = batch[1].view([batch_size * rotation])
                batch[2] = batch[2].view([batch_size * rotation])
                return batch
        
        else:
            # load function for supervised model mode, the trainer returns:
            # - the index for the image
            # - the categroy label for the image
            # - the image
            def _load_function(idx):
                idx = idx % len(self.dataset)
                img, category_label, index = self.dataset[idx]
                return img, category_label, index
            
            _collate_fun = default_collate
    
        tnt_dataset = tnt.dataset.ListDataset(elem_list = range(self.epoch_size),
                                              load = _load_function)
        data_loader = tnt.dataset.parralel(batch_size = self.batch_size,
                                           collate_fn = _collate_fun,
                                           num_workers = self.num_workers,
                                           shuffle = self.shuffle)

        return data_loader
    

    def __call__(self, epoch = 0):
        self.random_seed = epoch * self.epoch_size
        random.seed(self.random_seed)
        return self.data_loader_iter


    def __len__(self):
        return len(self.data_loader_iter)  