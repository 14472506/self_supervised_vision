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
from transforms import training_augmentations, setup_augmentations

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
        # setup
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # data location
        self.root = "./jersey_royals"
        
        # train dataset config
        self.train_batch_size = 2
        self.train_shuffle = True
        self.train_workers = 2
        self.train_augs = training_augmentations()

        # test dataset config
        self.test_batch_size = 1
        self.test_shuffle = False
        self.test_workers = 1
        self.setup_augs = setup_augmentations()

        # loading dataset
        self.load_dataset()

        # getting model
        self.model = resnet50_rotation_classifier(pre_trained=True, num_rotations=4)
        self.model.to(self.device)

        # optimizer config
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.optimizer_init()

        #loop config
        self.epochs = 10
        self.print_freque = 20
        self.loop_epochs = 10
        self.loop()


    def load_dataset(self, split_val = 0.8, transforms=True):
        """
        Detials
        """
        # get dataset
        if transforms:
            self.data_set = RotationDataset(root=self.root, transform=self.setup_augs)        
        else:
            self.data_set = RotationDataset(root=self.root)
        
        # get train test split values
        train = round(len(self.data_set)*0.8)
        test = len(self.data_set) - train

        # get train and test set
        self.train_set, self.test_set = torch.utils.data.random_split(self.data_set, [train, test])

        # get train and test loader
        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size = self.train_batch_size,
                                                        shuffle = self.train_shuffle,
                                                        num_workers = self.train_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size = self.test_batch_size,
                                                       shuffle = self.test_shuffle,
                                                       num_workers = self.test_workers)
    

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
            
            self.train_one_epoch(epoch)

            self.val_one_epoch()
        
        # training complete
        print("training finished")
    

    def train_one_epoch(self, epoch):
        """
        Details
        """
        # initialising running loss
        self.model.train()
        running_loss = 0.
        loss_acc = 0.
        for i, data in enumerate(self.train_loader, 0):
            # get data from loader
            input, labels = data
            input, labels = input.to(self.device), labels.to(self.device)
            
            # set param gradient to zero
            self.optimizer.zero_grad()
            
            # forward + backward + optimizer
            output = self.model(input)
            loss = classification_loss(output, labels)
            loss.backward()
            self.optimizer.step()
            
            loss_acc += loss.item()
            # printing stats
            running_loss += loss.item()
            if i % self.print_freque == self.print_freque-1: # printing every 200 minibatches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss/self.print_freque:.3f}")
                running_loss = 0
        
        print(loss_acc/len(self.train_loader))


    def val_one_epoch(self):
        """
        Detials
        """
        self.model.eval()
        loss_acc = 0.
        for i, data in enumerate(self.test_loader, 0):
            # get data from loader
            input, label = data
            input, label = input.to(self.device), label.to(self.device)

            with torch.no_grad():
                output = self.model(input) 
            loss = classification_loss(output, label)

            loss_acc += loss.item()

        print(loss_acc/len(self.test_loader))       