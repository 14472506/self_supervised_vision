"""
Detials
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
import torch 
import torchvision

from torch import Tensor, nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# =============================================================================================== #
# Augmentations
# =============================================================================================== #
def setup_augmentations(resize=(720, 1280)):
    """
    Detials
    """
    augs = A.Compose([
        A.Resize(*resize),
        A.Normalize(),
        ToTensorV2(),
        ])
    return augs

def training_augmentations(resize=(720, 1280), crop_size=(310, 426)):
    """
    Detials
    """
    augs = A.Compose([
        # standard
        A.Resize(*resize),
        A.RandomCrop(*crop_size),
        #A.Normalize(),

        # aditional augmentations
        A.ToGray(p=0.05),
        A.HorizontalFlip(p=0.5),

        ToTensorV2()
        ], p=1)
    return augs
    

def setup_augmentations(resize=(720, 1280)):
    """
    Detials
    """
    augs = A.Compose([
        A.Resize(*resize),
        A.Normalize(),
        ToTensorV2()
        ])
    return augs