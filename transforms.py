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

def training_augmentations(resize=(720, 1280), crop_size=(720, 1280)):
    """
    Detials
    """
    augs = A.Compose([
        A.Normalize(),
        A.OneOf([
            A.RandomCrop(*crop_size, p=0.33),
            A.Resize(*resize, p=0.66)
            ], p=1),
        A.Downscale (scale_min=0.25,
                     scale_max=0.25,
                     interpolation=None,
                     p=0.1),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2,
                                 sat_shift_limit= 0.2, 
                                 val_shift_limit=0.2, 
                                 p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                       contrast_limit=0.2,
                                       p=0.9),
            ],p=0.2),
        A.ToGray(p=0.1),
        ToTensorV2()
        ], p=1)
    return augs
