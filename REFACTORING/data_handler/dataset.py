"""
name        : dataset.py

task        : The script implements the task specific self supervised learning datasets. This is
              achieved through inheriting the the base dataset class and then applying the specific
              modifications to the data to generate the training and ground truth data for the task
              specific self supervised learning task. 

edited by   : bradley hurst
"""
# import 
from base import BaseDataset
from .utils import basic_square_crop, resize

import torchvision.transforms as torch_trans
import torch.nn.functional as torch_fun
import numpy as np
import os
from PIL import Image
import torch

# classes
class RotNetDataset(BaseDataset):
    """
    Detials
    """
    def __init__(self, root, seed=42, num_rotations=4):
        """
        Detials
        """
        super().__init__(root, seed)
        self.rotation_degrees = np.linspace(0, 360, num_rotations + 1).tolist()[:-1]

    def __getitem__(self, idx):
        """
        method_name : __getitem__

        task        : base method that returnes indexed image from dataset when called

        edited by   : bradley hurst
        """
        # load called RGB image 
        img_path = os.path.join(self.root, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        # getting basic image square
        image = basic_square_crop(image)
        image = resize(image)

        # converting image to tensor
        tensor_transform = torch_trans.Compose([torch_trans.ToTensor()])
        image_tensor = tensor_transform(image)

        # select random rotation
        theta = np.random.choice(self.rotation_degrees, size=1)[0]
        rotated_image_tensor = self.rotate_image(image_tensor.unsqueeze(0), theta).squeeze(0)
        label = torch.tensor(self.rotation_degrees.index(theta)).long()

        # returning rotated image tensor and label
        return rotated_image_tensor, label

    def rotate_image(self, image_tensor, theta):
        """
        Detials
        """
        # get tensor image data type
        dtype = image_tensor.dtype

        # covert degrees to radians and converting to tensor
        theta *= np.pi/180
        theta = torch.tensor(theta)

        # retrieveing rotation matrix around the z axis
        rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                        [torch.sin(theta), torch.cos(theta), 0]])
        rotation_matrix = rotation_matrix[None, ...].type(dtype).repeat(image_tensor.shape[0], 1, 1)
        
        # appling rotation
        grid = torch_fun.affine_grid(rotation_matrix,
                                     image_tensor.shape,
                                     align_corners=True).type(dtype)
        rotated_torch_image = torch_fun.grid_sample(image_tensor, grid, align_corners=True)

        # returning rotated image tensor
        return rotated_torch_image