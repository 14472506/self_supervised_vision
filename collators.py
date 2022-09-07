"""
Detials
"""
# =============================================================================================== #
# ----- Imports --------------------------------------------------------------------------------- #
# =============================================================================================== #
from multiprocessing.sharedctypes import Value
import random
import itertools
from tqdm import tqdm

import numpy as np
from scipy.spatial.distance import hamming

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.datasets import VOCSegmentation
from torchvision.transforms import transforms

# =============================================================================================== #
# ----- Classes --------------------------------------------------------------------------------- #
# =============================================================================================== #
class DataCollator(object):
    def __init__(self):
        pass


    def __call__(self, examples):
        pass


    def _preprocess_batch(self, examples):
        """
        Detials
        """
        examples = [(example[0], torch.tensor(example[1]).long()) for example in examples]
        if all(x.shape == examples[0][0].shape for x, y in examples):
            x, y = tuple(map(torch.stack, zip(*examples)))
            return x, y
        else:
            raise ValueError('Examples must contain the same shape!')


class RotationCollator(DataCollator):
    """
    Details
    """
    def __init__(self, num_rotations=4, rotation_procedure='all'):
        """
        Details
        """
        super(RotationCollator).__init__()
        self.num_rotations = num_rotations
        self.rotation_degrees = np.linspace(0, 360, num_rotations + 1).tolist()[:-1]
        
        assert rotation_procedure in ['all', 'random']
        self.rotation_procedure = rotation_procedure 
    
    
    def __call__(self, example):
        """
        Details
        """
        batch = self._preprocess_batch(example)
        x, y = batch    
        batch_size = x.shape[0]

        if self.rotation_procedure == 'random':
            for i in range(batch_size):
                theta = np.random.choice(self.rotation_degrees, size=1)[0]
                x[i] = self.rotate_image(x[i].unsqueeze(0), theta=theta).squeeze(0)
                y[i] = torch.tensor(self.rotation_degrees.index(theta)).long()

        elif self.rotation_procedure == 'all':
            x_, y_ = [], []
            for theta in self.rotation_degrees:
                x_.append(self.rotate_image(x.clone(), theta=theta))
                y_.append(torch.tensor(self.rotation_degrees.index(theta)).long().repeat(batch_size))
            
            x, y = torch.cat(x_), torch.cat(y_)
            permutation = torch.randperm(batch_size * 4)
            x, y = x[permutation], y[permutation]
        
        return x, y


    @staticmethod
    def get_rotation_matrix(theta, mode='degrees'):
        """
        Detials
        """
        assert mode in ['degrees', 'radians']

        if mode == 'degrees':
            theta *= np.pi/180
        
        theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                             [torch.sin(theta), torch.cos(theta), 0]])


    def rotate_image(self, x, theta):
        """
        Details
        """
        dtype = x.dtype
        rotation_matrix = self.get_rotation_matrix(theta=theta, mode='degrees')[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(rotation_matrix, x.shape, align_corners=True).type(dtype)
        x = F.grid_sample(x, grid, align_corners=True)
        return x