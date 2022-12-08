"""
Detials
"""
# imports 
from .dataset import RotNetDataset, JigsawDataset

import torch
import numpy as np
import random

# classes
class DataHandler():
    """
    Detials
    """
    def __init__(self, root, dataset_flag, seed=42, train_test_split=0.8, train_val_split=0.8):
        """
        Detials
        """
        # initialising torch generator and setting random seed
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

        # attributes
        self.root = root
        self.seed = seed
        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        self.dataset_flag = dataset_flag

    def load_dataset(self):
        """
        Detials
        """
        # getting base dataset
        if self.dataset_flag == "RotNet":
            base_dataset = RotNetDataset(self.root)
        elif self.dataset_flag == "Jigsaw":
            base_dataset = JigsawDataset(self.root, num_tiles=4, num_permutations=24)
        else:
            print("Dataset Not Specified")

        # splitting base dataset into train_base and test sets
        train_base_size = int(len(base_dataset)*self.train_test_split)
        test_size = len(base_dataset) - train_base_size
        train_base, test = torch.utils.data.random_split(base_dataset, [train_base_size, test_size])

        # spliting train_base into train and validations
        train_size = int(len(train_base)*self.train_val_split)
        validation_size = len(train_base) - train_size
        train, validation = torch.utils.data.random_split(train_base, [train_size, validation_size])

        # returing train, test, and validation datasets
        return train, test, validation

    def data_laoders(self):
        """
        Detials
        """
        # get train test and val datasets
        train, test, validation = self.load_dataset()

        # training data_loader
        train_loader = torch.utils.data.DataLoader(train,
                            batch_size = 8,
                            shuffle = True,
                            num_workers = 8,
                            worker_init_fn = self.seed_worker,
                            generator = self.gen)

        # testing data_laoder
        test_loader = torch.utils.data.DataLoader(test,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 1,
                            worker_init_fn = self.seed_worker,
                            generator = self.gen)

        # validation data_laoder
        validation_loader = torch.utils.data.DataLoader(validation,
                            batch_size = 1,
                            shuffle = False,
                            num_workers = 1,
                            worker_init_fn = self.seed_worker,
                            generator = self.gen)

        return train_loader, test_loader, validation_loader
        
    def seed_worker(self, worker_id):
        """
        Details
        """
        info = torch.utils.data.get_worker_info()
        worker_seed =  torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed) 
        random.seed(worker_seed)