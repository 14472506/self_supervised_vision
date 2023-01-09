"""
Detials
"""
# imports
import numpy as np
import random
import os
import torch

from data_handler import DataHandler
from model import rotnet_setup, jigsaw_setup, jigrot_setup
#from .loops import classification_training_loop, classification_validation_loop
from .loops import multiclass_training_loop, multiclass_validation_loop
from saver import model_saver
from logger import recorder_dict_multiclass, save_config, save_records

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
        #self.train_one_epoch = classification_training_loop
        #self.validate_one_epoch = classification_validation_loop
        self.train_one_epoch = multiclass_training_loop
        self.validate_one_epoch = multiclass_validation_loop

        # initialising saving
        self.recording = recorder_dict_multiclass()
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
        elif self.cd["model"]["name"] == "JigRot":
            model, optimiser, criterion = jigrot_setup(self.cd)
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
        # multi or single class model
        if self.cd["model"]["name"] == "JigRot":
            best_model_loss = 100
            best_model_l1_loss = 100
            best_model_l2_loss = 100
            best_model_comb = [100, 100]
            best_model_y1_acc = 0
            best_model_y2_acc = 0
            best_model_acc_comb = [0, 0]

        # looping through epochs
        for epoch in range(self.cd["loop"]["start_epoch"], self.cd["loop"]["end_epoch"], 1): 
            # train one epoch
            train_loop_data, self.count = self.train_one_epoch(epoch,
                                            self.count,
                                            self.model,
                                            self.train_loader, self.device, self.optimiser,
                                            self.criterion,
                                            self.print_freque)

            # val one epoch
            val_loop_data = self.validate_one_epoch(self.model,
                                            self.validation_loader,
                                            self.device,
                                            self.criterion)
            
            # recording loop results
            self.recording["epoch"].append(epoch)
            self.recording["y1_train_acc"].append(train_loop_data["y1_acc"])
            self.recording["y2_train_acc"].append(train_loop_data["y2_acc"])
            self.recording["y1_validation_acc"].append(val_loop_data["y1_acc"])
            self.recording["y2_validation_acc"].append(val_loop_data["y2_acc"])
            self.recording["training_tot_loss"].append(train_loop_data["total_loss"])
            self.recording["training_l1_loss"].append(train_loop_data["y1_loss"])
            self.recording["training_l2_loss"].append(train_loop_data["y2_loss"])
            self.recording["validation_tot_loss"].append(val_loop_data["total_loss"])
            self.recording["validation_l1_loss"].append(val_loop_data["y1_loss"])
            self.recording["validation_l2_loss"].append(val_loop_data["y2_loss"])

            model_saver("outputs/" + self.cd["logging"]["experiment_name"],
                 "last_model.pth",
                 epoch,
                 self.model, 
                 self.optimiser)

            #######################################################################################
            # ----- losses ------------------------------------------------------------------------
            if val_loop_data["total_loss"] < best_model_loss:
                self.recording["best_tot_epoch"].append(epoch)
                self.recording["best_tot_training_loss"].append(train_loop_data["total_loss"])
                self.recording["best_tot_validation_loss"].append(val_loop_data["total_loss"])
                model_saver("outputs/" + self.cd["logging"]["experiment_name"],
                     "best_tot_model.pth",
                     epoch,
                     self.model, 
                     self.optimiser)
                best_model_loss = val_loop_data["total_loss"]
            
            if val_loop_data["y1_loss"] < best_model_l1_loss:
                self.recording["best_l1_epoch"].append(epoch)
                self.recording["best_l1_training_loss"].append(train_loop_data["y1_loss"])
                self.recording["best_l1_validation_loss"].append(val_loop_data["y1_loss"])
                model_saver("outputs/" + self.cd["logging"]["experiment_name"],
                     "best_y1_model.pth",
                     epoch,
                     self.model, 
                     self.optimiser)
                best_model_l1_loss = val_loop_data["y1_loss"]

            if val_loop_data["y2_loss"] < best_model_l2_loss:
                self.recording["best_l2_epoch"].append(epoch)
                self.recording["best_l2_training_loss"].append(train_loop_data["y2_loss"])
                self.recording["best_l2_validation_loss"].append(val_loop_data["y2_loss"])
                model_saver("outputs/" + self.cd["logging"]["experiment_name"],
                     "best_y2_model.pth",
                     epoch,
                     self.model, 
                     self.optimiser)
                best_model_l2_loss = val_loop_data["y2_loss"]

            if val_loop_data["y1_loss"] < best_model_comb[0]:
                if val_loop_data["y1_loss"] < best_model_comb[1]:
                    self.recording["best_comb_epoch"].append(epoch)
                    self.recording["best_comb_train_loss"].append([train_loop_data["y1_loss"], train_loop_data["y2_loss"]])
                    self.recording["best_comb_val_loss"].append([val_loop_data["y1_loss"], val_loop_data["y2_loss"]])
                    model_saver("outputs/" + self.cd["logging"]["experiment_name"],
                         "best_comb_model.pth",
                         epoch,
                         self.model, 
                         self.optimiser)
                    best_model_comb[0] = val_loop_data["y1_loss"]
                    best_model_comb[1] = val_loop_data["y2_loss"]     

            # ----- Accuracies --------------------------------------------------------------------
            if val_loop_data["y1_acc"] > best_model_y1_acc:
                self.recording["best_y1_acc_epoch"].append(epoch)
                self.recording["best_y1_acc"].append(train_loop_data["y1_acc"])
                model_saver("outputs/" + self.cd["logging"]["experiment_name"],
                     "best_y1_acc_model.pth",
                     epoch,
                     self.model, 
                     self.optimiser)
                best_model_y1_acc = val_loop_data["y1_acc"]

            if val_loop_data["y2_acc"] > best_model_y2_acc:
                self.recording["best_y2_acc_epoch"].append(epoch)
                self.recording["best_y2_acc"].append(train_loop_data["y2_acc"])
                model_saver("outputs/" + self.cd["logging"]["experiment_name"],
                     "best_y2_acc_model.pth",
                     epoch,
                     self.model, 
                     self.optimiser)
                best_model_y2_acc = val_loop_data["y2_acc"]

            if val_loop_data["y1_acc"] > best_model_acc_comb[0]:
                if val_loop_data["y2_acc"] > best_model_acc_comb[1]:
                    self.recording["best_comb_acc_epoch"].append(epoch)
                    self.recording["best_comb_acc"].append([val_loop_data["y1_acc"], val_loop_data["y2_acc"]])
                    model_saver("outputs/" + self.cd["logging"]["experiment_name"],
                         "best_comb_acc_model.pth",
                         epoch,
                         self.model, 
                         self.optimiser)
                    best_model_comb_acc[0] = val_loop_data["y1_acc"]
                    best_model_comb_acc[1] = val_loop_data["y2_acc"]       
            
            
            #######################################################################################
            ## printing loop results
            print("training results: ", epoch_training_loss, "val results: ", epoch_validation_loss)

        # save records
        save_records(self.recording, self.cd)
        print("training complete")