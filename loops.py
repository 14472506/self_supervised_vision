"""
Details - something along the lines of this is the main loop excecution
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
# imports

# torch imports
from random import shuffle
import torch 
import torchvision.transforms as T
import torch.optim as optim

# model imports
from models import resnet50_rotation_classifier
from losses import classification_loss
from datasets import RotationDataset

# =============================================================================================== #
# Classes
# =============================================================================================== #
class Training_loop():
    """
    def
    """

    def __init__(self):
        """
        Details
        """
        # data location
        self.root = "./jersey_royals"
        
        # train dataset config
        self.train_batch_size = 4
        self.train_shuffle = True
        self.train_workers = 2

        # test dataset config
        self.test_batch_size = 1
        self.test_shuffle = False
        self.test_workers = 1

        # loading dataset
        self.load_dataset(train=True)

        # getting model
        self.model = resnet50_rotation_classifier(pre_trained=True, num_rotations=4)

        # getting loss function
        self.loss = classification_loss()

        # optimizer config
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.optimizer_init()

        #loop config
        self.loop_epochs = 10


    def load_dataset(self, train=False, validation=False, test=False):
        """
        Detials
        """
        if train == True:
            self.train_set = RotationDataset(root=self.root)
            self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                            batch_size = self.train_batch_size,
                                                            shuffle = self.train_shuffle,
                                                            num_workers = self.train_workers)
        if test == True:
            pass
        if validation == True:
            pass

    
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
        for epoch in range(self.epochs):
            
            # initialising running loss
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                
                # get data from loader
                input, labels = data

                # set param gradient to zero
                self.optimizer.zero_grad()

                # forward + backward + optimizer
                output = self.model(input)
                loss = self.loss(output, labels)
                loss.backward()
                self.optimizer.step()

                # printing stats
                running_loss += loss.item()
                if i % 2000 == 1999: # printing every 200 minibatches
                    print(f"[{epoch + 1}, {i +  1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0
        
        # training complete
        print("training finished")