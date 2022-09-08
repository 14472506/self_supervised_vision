"""
Detials:  Something along the lines of the role of this file is to provide the self supervised
          Datasets for the learning process
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
import numpy as np
import os
from PIL import Image

#import torchnet as tnt
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torch.nn.functional as F

# =============================================================================================== #
# Classes
# =============================================================================================== #
class RotationDataset(data.Dataset):
    """
    detials of class
    """
    def __init__(self, root, num_rotations=4, split=None, transform=None):
        """
        Detials on init
        """
        self.root = os.path.expanduser(root)
        self.image_files = []
        for file in os.listdir(self.root):
            self.image_files.append(file)

        self.rotation_degrees = np.linspace(0, 360, num_rotations + 1).tolist()[:-1]
    

    def __getitem__(self, idx):
        """
        Detials
        """
        image_path =  os.path.join(self.root, self.image_files[idx])
        img = Image.open(image_path).convert("RGB")

        # further augmentation capability here

        transform = T.Compose([T.ToTensor()])
        torch_img = transform(img)

        theta = np.random.choice(self.rotation_degrees, size=1)[0]
        out_img = self.rotate_image(torch_img.unsqueeze(0), theta=theta).squeeze(0)
        label = torch.tensor(self.rotation_degrees.index(theta)).long()
        
        return out_img, label
    

    def __len__(self):
        """
        Details
        """
        return len(self.image_files)


    def rotate_image(self, x, theta):
        """
        Details
        """
        dtype = x.dtype
        rotation_matrix = self.get_rotation_matrix(theta=theta, mode='degrees')[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(rotation_matrix, x.shape, align_corners=True).type(dtype)
        x = F.grid_sample(x, grid, align_corners=True)
        return x
    

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


#if __name__ == "__main__":
#    Rot = RotationDataset("/home/bradley/workspace/self_supervised_vision/jersey_royals")
#    x, y = Rot.__getitem__(0)
#    print(x, y)