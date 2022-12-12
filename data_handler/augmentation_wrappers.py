"""
Detials
"""
# imports
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision.transforms as T
import numpy as np

# Augmentation functions
def rotation_test_augmentations():
    """
    Detials
    """
    augs = A.Compose([
        A.Normalize(),
        ToTensorV2(),
        ])
    return augs

def rotation_train_augmentations():
    """
    Detials
    """
    augs = A.Compose([
        A.Normalize(),
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
        A.ToGray(p=0.2),
        ToTensorV2()
        ], p=1)
    return augs

def jigsaw_test_augmentations():
    """
    Detials
    """
    augs = A.Compose([
        A.Normalize(),
        ToTensorV2(),
        ], p=1)
    return augs

def jigsaw_train_augmentations():
    """
    Detials
    """
    augs = A.Compose([
        A.Normalize(),
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
        A.ToGray(p=0.2),
        ToTensorV2()
        ], p=1)
    return augs

# wrapper classes
class RotNetWrapper(torch.utils.data.Dataset):
    """
    Detials
    """
    def __init__(self, dataset, transforms):
        """
        Detials
        """
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Detials
        """
        image, label = self.dataset[idx]

        pil_trans = T.ToPILImage()
        pil = pil_trans(image)
        np_img = np.array(pil)
        transformed = self.transforms(image=np_img)['image']
     
        return(transformed, label)

    def __len__(self):
        """
        Details
        """
        return len(self.dataset)

class JigsawWrapper(torch.utils.data.Dataset):
    """
    Detials
    """
    def __init__(self, dataset, transforms):
        """
        Detials
        """
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Detials
        """
        # getting image
        image, label = self.dataset[idx]
        
        # prepare augmented image stack
        aug_stack = []
        
        # loop through base stack
        for i in image:
            pil_trans = T.ToPILImage()
            pil = pil_trans(i)
            np_img = np.array(pil)
            transformed = self.transforms(image=np_img)["image"]
            aug_stack.append(transformed)

        stack = torch.stack(aug_stack)
        image = stack

        return(image, label)

    def __len__(self):
        """
        Details
        """
        return len(self.dataset)