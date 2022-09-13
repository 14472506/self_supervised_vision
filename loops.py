"""
Details - something along the lines of this is the main loop excecution
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
# imports
import random
from random import shuffle
import numpy as np
import os

# torch imports
import torch 
import torchvision.transforms as T
import torch.optim as optim

# model imports
from models import resnet50_rotation_classifier
from losses import classification_loss
from datasets import RotationDataset, TrainAugMapper
from transforms import training_augmentations, setup_augmentations
from utils import make_dir, model_saver

# =============================================================================================== #
# Classes
# =============================================================================================== #
class Training_loop():
    """
    def
    """

    def __init__(self, cd, seed=42):
        """
        Details
        """
        # ------ training loop setup attributes ------------------------------------------------- #
        # setup
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.seed = seed

        # experiment data setup
        self.experiment_name = cd['EXPERIMENT_NAME']
        self.exp_dir = "outputs/" + self.experiment_name
        make_dir(self.exp_dir)

        # data location
        self.root = cd['DATASET']['ROOT']
        
        # train dataset config
        self.train_batch_size = cd['DATASET']['TRAIN']['BATCH_SIZE']
        self.train_shuffle = cd['DATASET']['TRAIN']['SHUFFLE']
        self.train_workers = cd['DATASET']['TRAIN']['WORKERS']
        self.train_augs = training_augmentations()

        # test dataset config
        self.val_batch_size = cd['DATASET']['TEST']['BATCH_SIZE']
        self.val_shuffle = cd['DATASET']['TEST']['SHUFFLE']
        self.val_workers = cd['DATASET']['TEST']['WORKERS']
        self.setup_augs = setup_augmentations()

        # ----- training loop methods config and call ------------------------------------------- #
        # set random seed
        self.set_seed()

        # loading dataset
        self.load_dataset()

        # getting model
        self.model = resnet50_rotation_classifier(pre_trained=cd['MODEL']['PRE_TRAINED'],
                                                  num_rotations=cd['MODEL']['NUM_ROTATIONS'])
        self.model.to(self.device)

        # optimizer config
        self.learning_rate = cd['OPTIMIZER']['LEARNING_RATE']
        self.momentum = cd['OPTIMIZER']['MOMENTUM']
        self.optimizer_init()

        #loop config
        self.start_epoch = cd['LOOP']['STARTING_EPOCH']
        self.epochs = cd['LOOP']['EPOCHS']
        self.print_freque = cd['LOOP']['PRINT_FREQ']
        self.loop()


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


    def load_dataset(self, seed=42, split_percentage=0.8):
        """
        Detials
        """
        # define the worker init function
        def seed_worker(worker_id):
            """
            Details
            """
            info = torch.utils.data.get_worker_info()
            worker_seed =  torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed) 
            random.seed(worker_seed)
            #print("Worker ID:", info.id, "Worker Seed:",worker_seed)

        # setting random seed
        gen = torch.Generator()
        gen.manual_seed(seed)

        # get base dataset
        self.base_set = RotationDataset(root=self.root, transforms=self.setup_augs, seed=self.seed)
        
        # get train and validation dataset
        train_size = int(len(self.base_set)*split_percentage)
        val_size = len(self.base_set) - train_size
        train_base, self.val_set = torch.utils.data.random_split(self.base_set, [train_size, val_size])
        self.train_set = TrainAugMapper(train_base, self.train_augs)
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size = self.train_batch_size,
                                                        shuffle = self.train_shuffle,
                                                        num_workers = self.train_workers,
                                                        worker_init_fn = seed_worker,
                                                        generator = gen)
        
        self.val_loader = torch.utils.data.DataLoader(self.val_set,
                                                        batch_size = self.val_batch_size,
                                                        shuffle = self.val_shuffle,
                                                        num_workers = self.val_workers,
                                                        worker_init_fn = seed_worker,
                                                        generator = gen)

    
    def optimizer_init(self):
        """
        Details
        """
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr = self.momentum)


    def loop(self):
        """
        Detials
        """
        # best model tracker initialised and arbitrary high val 
        best_model = 100 

        # loop through epoch range
        for epoch in range(self.start_epoch, self.epochs, 1):
            
            # run training on one epoch
            epoch_train_loss = self.train_one_epoch(epoch)

            # run validation on one one epoch
            epoch_val_loss = self.val_one_epoch()

            # save last model 
            model_saver(epoch, self.model, self.optimizer, self.exp_dir, "last_model.pth")

            # saving best model
            if epoch_val_loss < best_model:
                model_saver(epoch, self.model, self.optimizer, self.exp_dir, "best_model.pth") 

            # carry out model saving
            print("training results: ", epoch_train_loss, "val results: ", epoch_val_loss)

        # training complete
        print("training finished")

    
    def train_one_epoch(self, epoch):
        """
        Detials
        """
        # set model to train
        self.model.train()

        # init train loss accumulator and running loss for print
        acc_loss = 0
        running_loss = 0

        # loop through train loader
        for i, data in enumerate(self.train_loader, 0):
                
            # get data from loader and send it to device
            input, labels = data
            input = input.float()
            input, labels = input.to(self.device), labels.to(self.device)

            # set param gradient to zero
            self.optimizer.zero_grad()

            # forward + backward + optimizer
            output = self.model(input)
            loss = classification_loss(output, labels)
            loss.backward()
            self.optimizer.step()

            # printing stats
            running_loss += loss.item()
            acc_loss += loss.item()
            if i % self.print_freque == self.print_freque-1:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss/self.print_freque:.3f}")
                running_loss = 0
        
        # calculating and returning trainig loss over epoch
        epoch_loss = acc_loss/len(self.train_loader)
        return epoch_loss
    

    def val_one_epoch(self):
        """
        Detials
        """
        # set model to validation
        self.model.eval()

        # init validation loss accumulator
        loss_acc = 0

        # loop through validation loader
        for i, data in enumerate(self.val_loader, 0):
            
            # get data and send it to device 
            input, label = data
            input, label = input.to(self.device), label.to(self.device)

            # with torch no grad get output from model
            with torch.no_grad():
                output = self.model(input) 
            
            # get loss and add it to loss accumulator
            loss = classification_loss(output, label)
            loss_acc += loss.item()

        # calculate and return validation loss
        val_loss = loss_acc/len(self.val_loader)
        return val_loss