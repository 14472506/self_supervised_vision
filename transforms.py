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
def training_augmentations(resize=(720, 1280), crop_size=(310, 426)):
    """
    Detials
    """
    augs = A.Compose([
        # standard
        A.Resize(*resize),
        A.RandomCrop(*crop_size),
        A.Normalize(),

        # aditional augmentations
        A.ToGray(p=0.05),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
        ToTensorV2()
        ])
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