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
import json
import matplotlib.pyplot as plt

# torch imports
import torch 
import torchvision.transforms as T
import torch.optim as optim

# model imports
from models import resnet50_rotation_classifier, JigsawClassifier, ContextEncoder
from losses import classification_loss, reconstruction_loss
from datasets import RotationDataset, JigsawTrainAugMapper, JigsawDataset, ContextEncDataset, ContextEncTrainAugMapper
from transforms import jigsaw_training_augmentations, jigsaw_setup_augmentations
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
        self.cd = cd
        
        # setup
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.seed = seed
        self.set_seed()

        # experiment data setup
        self.experiment_name = self.cd['EXPERIMENT_NAME']
        self.exp_dir = "outputs/" + self.experiment_name
        make_dir(self.exp_dir)

        # data location
        self.root = self.cd['DATASET']['ROOT']
        
        # ----- train dataset config
        self.train_batch_size = self.cd['DATASET']['TRAIN']['BATCH_SIZE']
        self.train_shuffle = self.cd['DATASET']['TRAIN']['SHUFFLE']
        self.train_workers = self.cd['DATASET']['TRAIN']['WORKERS']
        self.min = self.cd['TRANSFORMS']['MIN']
        self.max = self.cd['TRANSFORMS']['MAX']
        self.crop_min = self.cd['TRANSFORMS']['CROP_MIN']
        self.crop_max = self.cd['TRANSFORMS']['CROP_MAX']
        
        self.train_augs = jigsaw_training_augmentations(resize=(self.min, self.max), crop_size=(self.crop_min, self.crop_max))

        # ----- test dataset config
        self.test_batch_size = self.cd['DATASET']['TEST']['BATCH_SIZE']
        self.test_shuffle = self.cd['DATASET']['TEST']['SHUFFLE']
        self.test_workers = self.cd['DATASET']['TEST']['WORKERS']

        self.val_batch_size = self.cd['DATASET']['VAL']['BATCH_SIZE']
        self.val_shuffle = self.cd['DATASET']['VAL']['SHUFFLE']
        self.val_workers = self.cd['DATASET']['VAL']['WORKERS']

        self.setup_augs = jigsaw_setup_augmentations(resize=(self.min, self.max))

        # loading dataset
        self.load_dataset()

        # getting model
        self.model = JigsawClassifier(pre_trained=cd['MODEL']['PRE_TRAINED'],
                                      num_tiles=9,
                                      num_permutations=100)#,
        #                            #(num_rotations=cd['MODEL']['NUM_ROTATIONS'])
        #self.model = ContextEncoder(target_size=(1080, 1080))
        self.model.to(self.device)

        # optimizer config
        self.optimizer_init()

        # init recorder
        self.init_recorder()

        #loop config
        self.save_config()
        self.start_epoch = cd['LOOP']['STARTING_EPOCH']
        self.epochs = cd['LOOP']['EPOCHS']
        self.print_freque = cd['LOOP']['PRINT_FREQ']
        self.loop()

        #self.eval()


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


    def load_dataset(self, seed=42, split_percentage=0.8, val_split=0.8):
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
        self.base_set = RotationDataset(root=self.root, seed=self.seed)
        #self.base_set = JigsawDataset(root=self.root,
        #                              num_tiles=9,
        #                              num_permutations=100)
        #self.base_set = ContextEncDataset(root=self.root, mask_method='random_blocks') 
        
        # get train and validation dataset
        train_size = int(len(self.base_set)*split_percentage)
        test_size = len(self.base_set) - train_size
        train_to_split, test_base = torch.utils.data.random_split(self.base_set, [train_size, test_size])
        
        train_size = int(len(train_to_split)*val_split)
        val_size = len(train_to_split) - train_size
        train_base, val_base = torch.utils.data.random_split(train_to_split, [train_size, val_size]) 

        #self.train_set = train_base
        #self.val_set = val_base
        #self.test_set = test_base

        self.train_set = JigsawTrainAugMapper(train_base, self.train_augs)
        self.val_set = JigsawTrainAugMapper(val_base, self.setup_augs)
        self.test_set = JigsawTrainAugMapper(test_base, self.setup_augs)
        
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
        
        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                batch_size = self.cd["DATASET"]["TEST"]["BATCH_SIZE"],
                                                shuffle = self.cd["DATASET"]["TEST"]["SHUFFLE"],
                                                num_workers = self.cd["DATASET"]["TEST"]["WORKERS"],
                                                worker_init_fn = seed_worker,
                                                generator = gen)
    
        print(len(self.train_loader))
    def optimizer_init(self):
        """
        Details
        """
        if self.cd["OPTIMIZER"]["NAME"] == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(),
                                       lr = self.cd["OPTIMIZER"]["PARAMS"][0])
        elif self.cd["OPTIMIZER"]["NAME"] == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr = self.cd["OPTIMIZER"]["PARAMS"][0],
                                       momentum = self.cd["OPTIMIZER"]["PARAMS"][1],
                                       weight_decay = self.cd["OPTIMIZER"]["PARAMS"][2])
        else:
            print("optimizer not recognised")

    
    def init_recorder(self):
        self.data_recorder = {
            "train_loss": [],
            "val_loss"  : [],
            "best_loss" : [],
            "best_epoch": [],
        } 


    def loop(self):
        """
        Detials
        """
        # best model tracker initialised and arbitrary high val 
        best_model = 100 

        iter_count = 0
        # loop through epoch range
        for epoch in range(self.start_epoch, self.epochs, 1):
            
            # run training on one epoch
            epoch_train_loss, iter_count = self.train_one_epoch(epoch, iter_count)

            # run validation on one one epoch
            epoch_val_loss = self.val_one_epoch()
            
            # save last model 
            model_saver(epoch, self.model, self.optimizer, self.exp_dir, "last_model.pth")
            self.data_recorder["train_loss"].append(epoch_train_loss)
            self.data_recorder["val_loss"].append(epoch_val_loss)
            
            # saving best model
            if epoch_val_loss < best_model:
                model_saver(epoch, self.model, self.optimizer, self.exp_dir, "best_model.pth") 
                self.data_recorder["best_loss"].append(epoch_val_loss)
                self.data_recorder["best_epoch"].append(epoch)
                best_model = epoch_val_loss
            
            # carry out model saving
            print("training results: ", epoch_train_loss, "val results: ", epoch_val_loss)

        save_file = self.exp_dir + "/training_data.json"
        with open(save_file, "w") as f:
            json.dump(self.data_recorder, f)

        # training complete
        print("training finished")

    
    def train_one_epoch(self, epoch, iter_count):
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
            x, y = data
            #input = input.float()
            x, y = x.to(self.device), y.to(self.device)
            print(y)

            # set param gradient to zero
            self.optimizer.zero_grad()

            # forward + backward + optimizer
            output = self.model(x)
            #if i % 340 == 339:
            #    plt.imshow(output[0].cpu().detach().permute(1, 2, 0))
            #    plt.show()
            #print(output)
            #loss = classification_loss(output, labels)
            loss = reconstruction_loss(output, y)

            loss.backward()
            self.optimizer.step()

            # printing stats
            running_loss += loss.item()
            acc_loss += loss.item()
            if i % self.print_freque == self.print_freque-1:
            
                # get GPU memory usage
                mem_all = torch.cuda.memory_allocated(self.device) / 1024**3 
                mem_res = torch.cuda.memory_reserved(self.device) / 1024**3 
                mem = mem_res + mem_all
                mem = round(mem, 2)
                print("[epoch: %s][iter: %s][memory use: %sGB] total_loss: %s" %(epoch, iter_count, mem, running_loss/self.print_freque))

                running_loss = 0

            iter_count += 1

        
        # calculating and returning trainig loss over epoch
        epoch_loss = acc_loss/len(self.train_loader)
        return epoch_loss, iter_count
    

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
            #inputs, labels = data
            #inputs, labels = inputs.to(self.device), labels.to(self.device)

            # get data from loader and send it to device
            x, y = data
            #input = input.float()
            x, y = x.to(self.device), y.to(self.device)
            #print(labels)

            # with torch no grad get output from model
            with torch.no_grad():
                output = self.model(x) 
            
            # get loss and add it to loss accumulator
            #loss = classification_loss(output, labels)
            loss = classification_loss(output, y)
            loss_acc += loss.item()

        # calculate and return validation loss
        val_loss = loss_acc/len(self.val_loader)
        return val_loss
    

    def eval(self):
        """
        Detials
        """

        # load best model
        model_dir = self.exp_dir + "/best_model.pth"
        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint["state_dict"])

        # set model to eval
        self.model.eval()

        # init data collector
        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                   30, 31, 32, 33, 34, 35, 36, 27, 28, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                   70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                   80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                   90, 91, 92, 93, 94, 95, 96, 97, 98, 99)
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname:0 for classname in classes}
        results_rec = {classname:0 for classname in classes}

        # run eval loop
        with torch.no_grad():
            for data in self.test_loader:
                # get data
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # get predictions
                outputs = self.model(inputs)
                _, predictions = torch.max(outputs, 1)
                
                # getting collecting correct predictions
                for label, prediction in zip(labels, predictions):
                
                    label_val = (label == 1).nonzero(as_tuple=True)[0].item()
                    pred_val = prediction.item()

                    if label_val == pred_val:
                        correct_pred[classes[label_val]] += 1
                    total_pred[classes[label_val]] += 1

        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            results_rec[classname] = accuracy
            print('Accuracy for class: %s is %s' %(classname, accuracy))
        
        test_data = {
            "correct_preds"    : correct_pred,
            "total_preds"      : total_pred,
            "accuracy_results" : results_rec
            } 

        # saving data in json
        save_file = self.exp_dir + "/test_data.json"
        with open(save_file, "w") as f:
            json.dump(test_data, f)


    def save_config(self):
        """
        Detials
        """
        # saving data in json
        save_file = self.exp_dir + "/config.json"
        with open(save_file, "w") as f:
            json.dump(self.cd, f)