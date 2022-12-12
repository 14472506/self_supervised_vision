"""
name    : base_dataset.py

purpose : implements a base class that is inherited by other dataset classes for task specific
          self supervised data loaders. the base class loads the imagas into the dataset and
          the base structure.

edit by : bradley hurst
"""
# imports
import os
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image

# base class 
class BaseDataset(data.Dataset):
    """
    class_name  : BaseDataSet

    task        : acts as the base dataset to be inherited by task specific dataset classes for 
                  self supervised learning tasks. The base class simply loads the images data
                  from the base dir and returnes an RGB image loaded with PIL when provided with
                  and index through the get item method. 

    edited by   : bradley hurst
    """
    def __init__(self, root, seed=42):
        """
        method_name : __init__

        task        : initialises the attributes of the base dataset class. It provides the data
                      root, and images list for indexing. it also sets the random seed for numpy
                      to make the pipeline deterministic.

        edited by   : bradley hurst
        """
        # retrieveing image data from root directory
        self.root = os.path.expanduser(root)
        self.images = []
        for image in os.listdir(self.root):
            self.images.append(image)

        # setting numpy random seed
        np.random.seed(seed)
    
    def __len__(self):
        """
        method_name : __len__

        task        : base method that returns the count of images in the dataset

        edited by   : bradley hurst 
        """
        return len(self.images)