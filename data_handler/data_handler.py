"""
Detials
"""
# imports 
from .dataset import RotNetDataset, JigsawDataset, JigRotDataset
from .augmentation_wrappers import jigsaw_test_augmentations, jigsaw_train_augmentations, JigsawWrapper, rotation_test_augmentations, rotation_train_augmentations, RotNetWrapper
import torch
import numpy as np
import random

# classes
class DataHandler():
    """
    Detials
    """
    def __init__(self, conf_dict, seed=42):
        """
        Detials
        """
        # initialising torch generator and setting random seed
        self.cd = conf_dict
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

        # attributes
        self.train_test_split = self.cd["data"]["tt_split"]
        self.train_val_split = self.cd["data"]["tv_split"]

    def load_dataset(self):
        """
        Detials
        """
        # getting base dataset
        if self.cd["model"]["name"] == "RotNet":
            base_dataset = RotNetDataset(self.cd["data"]["path"])
        elif self.cd["model"]["name"] == "Jigsaw":
            base_dataset = JigsawDataset(self.cd["data"]["path"],
                num_tiles=self.cd["model"]["num_tiles"],
                num_permutations=self.cd["model"]["permutations"])
        elif self.cd["model"]["name"] == "JigRot":
            base_dataset = JigRotDataset(self.cd["data"]["path"],
                num_tiles=self.cd["model"]["num_tiles"],
                num_perms=self.cd["model"]["num_perms"],
                num_rotations=self.cd["model"]["rotations"],
                tile_rotations=self.cd["model"]["tile_rotations"]
                )
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

            # getting base dataset
        if self.cd["model"]["name"] == "RotNet":
            train = RotNetWrapper(train, rotation_train_augmentations())
            test = RotNetWrapper(test, rotation_test_augmentations())
            validation =  RotNetWrapper(validation, rotation_test_augmentations()) 
        elif self.cd["model"]["name"] == "Jigsaw":
            train = JigsawWrapper(train, jigsaw_train_augmentations())
            test = JigsawWrapper(test, jigsaw_test_augmentations())
            validation =  JigsawWrapper(validation, jigsaw_test_augmentations()) 
        else:
            print("Dataset Not Specified")

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
                            batch_size = self.cd["data"]["train_loader"]["batch_size"],
                            shuffle = self.cd["data"]["train_loader"]["shuffle"],
                            num_workers = self.cd["data"]["train_loader"]["num_workers"],
                            worker_init_fn = self.seed_worker,
                            generator = self.gen)

        # testing data_laoder
        test_loader = torch.utils.data.DataLoader(test,
                            batch_size = self.cd["data"]["test_loader"]["batch_size"],
                            shuffle = self.cd["data"]["test_loader"]["shuffle"],
                            num_workers = self.cd["data"]["test_loader"]["num_workers"],
                            worker_init_fn = self.seed_worker,
                            generator = self.gen)

        # validation data_laoder
        validation_loader = torch.utils.data.DataLoader(validation,
                            batch_size = self.cd["data"]["validation_loader"]["batch_size"],
                            shuffle = self.cd["data"]["validation_loader"]["shuffle"],
                            num_workers = self.cd["data"]["validation_loader"]["num_workers"],
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