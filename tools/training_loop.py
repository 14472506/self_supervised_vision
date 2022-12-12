"""
Detials
"""
# imports
import numpy as np
import random
import os
import torch

from data_handler import DataHandler
from model import rotnet_setup, jigsaw_setup
from .loops import classification_training_loop, classification_validation_loop
from saver import model_saver
from logger import recorder_dict, save_config, save_records

# training class
class TrainingLoop():
    """
    Detials
    """
    def __init__(self, conf_dict, seed=42):
        """
        Detials
        """
        # initial config
        self.cd = conf_dict
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.seed = seed
        self.set_seed()

        # get data_loaders
        self.data_loader()

        # get model, optimiser, and criterion
        self.model_loader()

        # loop loging attributes
        self.count = 0
        self.print_freque = self.cd["logging"]["print_freque"]

        # loading training loops
        self.train_one_epoch = classification_training_loop
        self.validate_one_epoch = classification_validation_loop

        # initialising saving
        self.recording = recorder_dict()
        save_config(self.cd)

    def data_loader(self):
        """
        Detials
        """
        h = DataHandler(self.cd)
        train, _, val = h.data_laoders()
        self.train_loader = train
        self.validation_loader = val
    
    def model_loader(self):
        """
        Detials
        """
        if self.cd["model"]["name"] == "RotNet":
            model, optimiser, criterion = rotnet_setup(self.cd)
        elif self.cd["model"]["name"] == "Jigsaw":
            model, optimiser, criterion = jigsaw_setup(self.cd)
        else:
            print("Model Not Specified")

        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.model.to(self.device)

    def set_seed(self):
        """
        Details
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(self.seed)

    def loop(self):
        """
        Detials
        """
        # init counts
        best_model = 100

        # looping through epochs
        for epoch in range(self.cd["loop"]["start_epoch"], self.cd["loop"]["end_epoch"], 1):
            # train one epoch
            epoch_training_loss, self.count = self.train_one_epoch(epoch, self.count, self.model,
                                                    self.train_loader, self.device, self.optimiser,
                                                    self.criterion,
                                                    self.print_freque)

            # val one epoch
            epoch_validation_loss = self.validate_one_epoch(self.model,
                                    self.validation_loader,
                                    self.device,
                                    self.criterion)
            
            # recording loop results)
            self.recording["epoch"].append(epoch)
            self.recording["training_loss"].append(epoch_training_loss)
            self.recording["validation_loss"].append(epoch_validation_loss)
            model_saver("outputs/" + self.cd["logging"]["experiment_name"],
                 "last_model.pth",
                 epoch,
                 self.model, 
                 self.optimiser)

            # recording best results 
            if epoch_validation_loss < best_model:
                self.recording["best_epoch"].append(epoch)
                self.recording["best_training_loss"].append(epoch_training_loss)
                self.recording["best_validation_loss"].append(epoch_validation_loss)
                model_saver("outputs/" + self.cd["logging"]["experiment_name"],
                     "best_model.pth",
                     epoch,
                     self.model, 
                     self.optimiser)
                best_model = epoch_validation_loss
            
            # printing loop results
            print("training results: ", epoch_training_loss, "val results: ", epoch_validation_loss)

        # save records
        save_records(self.recording, self.cd)
        print("training complete")