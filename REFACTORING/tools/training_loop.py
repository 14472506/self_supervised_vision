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
from model import classification_training_loop, classification_validation_loop

# training class
class TrainingLoop():
    """
    Detials
    """
    def __init__(self, conf_dict={}, seed=42):
        """
        Detials
        """
        # initial config
        self.conf_dict = conf_dict
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.seed = seed
        self.set_seed()

        # get data_loaders
        self.data_loader()

        # get model, optimiser, and criterion
        self.model_loader("Jigsaw")

        # loop loging attributes
        self.count = 0
        self.print_freque = 20

        self.train_one_epoch = classification_training_loop
        self.validate_one_epoch = classification_validation_loop

    def data_loader(self):
        """
        Detials
        """
        h = DataHandler("data/jersey_royals_ssl_ds", "Jigsaw")
        train, _, val = h.data_laoders()
        self.train_loader = train
        self.validation_loader = val
    
    def model_loader(self, model_flag):
        """
        Detials
        """
        if model_flag == "RotNet":
            model, optimiser, criterion = rotnet_setup()
        elif model_flag == "Jigsaw":
            model, optimiser, criterion = jigsaw_setup(num_tiles=4, num_permutations=24)
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
        for epoch in range(0, 40, 1):
            # train one epoch
            epoch_training_loss, self.count = self.train_one_epoch(epoch, self.count, self.model,
                                                    self.train_loader, self.device, self.optimiser,
                                                    self.criterion,
                                                    self.print_freque)

            # val one epoch
            epoch_val_loss = self.validate_one_epoch(self.model,
                                    self.validation_loader,
                                    self.device,
                                    self.criterion)
