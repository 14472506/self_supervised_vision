"""
Detials
"""
# imports
import numpy as np
import random
import os
import torch

from data_handler import DataHandler
from model import rotnet_setup

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
        self.model_loader()

    def data_loader(self):
        """
        Detials
        """
        h = DataHandler("data/jersey_royals_ssl_ds")
        train, _, val = h.data_laoders()
        self.train_loader = train
        self.validation_laoder = val
    
    def model_loader(self):
        """
        Detials
        """
        model, optimiser, criterion = rotnet_setup()
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
        iter_count = 0

        # looping through epochs
        for epoch in range(0, 40, 1):
            # train one epoch
            self.train_one_epoch()

            # val one epoch
            self.validate_one_epoch()

    def train_one_epoch(self):
        """
        Detials
        """
        # model config
        self.model.train()

        # accume init here
        for i, data in enumerate(self.train_loader, 0):
            # extact data 
            x, y_gt = data
            x, y_gt = x.to(self.device), y_gt.to(self.device)

            # reset gradient
            self.optimiser.zero_grad

            # forward + backward + optimizer_step
            y_pred = self.model(x)
            loss = self.classification_loss(y_pred, y_gt)
            loss.backward()
            self.optimiser.step() 

            # reporting goes here
            print(loss.item())
        
    def validate_one_epoch(self):
        """
        Detials
        """
        # model config
        self.model.eval()

        # logging here

        for i, data in enumerate(self.validation_laoder, 0):
            # extract data
            x, y_gt = data
            x, y_gt = x.to(self.device), y_gt.to(self.device)

            # forwards 
            with torch.no_grad():
                y_pred = self.model(x)
            
            # get loss
            loss = classification_loss(y_pred, y_gt)

            print(loss.item())

        



