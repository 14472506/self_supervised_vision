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
import itertools
import tqmd

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
    def __init__(self, root, num_rotations=4, transforms=None, seed=42):
        """
        Detials on init
        """
        self.root = os.path.expanduser(root)
        self.image_files = []
        for file in os.listdir(self.root):
            self.image_files.append(file)

        self.transforms = transforms
        self.rotation_degrees = np.linspace(0, 360, num_rotations + 1).tolist()[:-1]
        np.random.seed(seed)


    def __getitem__(self, idx):
        """
        Detials
        """
        image_path =  os.path.join(self.root, self.image_files[idx])
        img = Image.open(image_path).convert("RGB")

        # further augmentation capability here
        if self.transforms:
            np_img = np.array(img)
            transformed = self.transforms(image=np_img)
            torch_img = transformed['image']
        else:
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


class TrainAugMapper(torch.utils.data.Dataset):
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
        transformed = self.transforms(image=np_img)
        image = transformed['image']
     
        return(image, label)


    def __len__(self):
        """
        Details
        """
        return len(self.dataset)

#if __name__ == "__main__":
#    Rot = RotationDataset("/home/bradley/workspace/self_supervised_vision/jersey_royals")
#    x, y = Rot.__getitem__(0)
#    print(x, y)


class JigsawDataset(data.Datasets):
    """
    Detials
    """

    def __init__(self,  num_tiles=9, num_permutations=1000, permgen_method='maximal',
                 grayscale_probability=0.3, buffer=True, jitter=True, normalization=True):
        """
        Detials
        """
        self.num_tiles = num_tiles
        self.num_permutations = num_permutations

    def __call__(self, examples):
        """
        Detials
        """
        batch =  self._preprocess_batch(examples)
        x, _ = batch
        
        # will need to address this
        assert img_height == img_width

        # gray scale goes here if gray scale is applied
        # GRAY SCALE

        # compute tile lengths and exract tiles
        tiles = []
        num_tiles_per_dimension = int(np.sqrt(self.num_tiles))
        tile_lenght = img_width // num_permutations

        buffer = int(tile_lenght * 0.1)

        # entering set of loops to go through both dimensions of image
        for i in range(num_tiles_per_dimension):
            for j in range(num_tiles_per_dimension):
                
                if self.buffer:
                    tile_ij = torch.empty(batch_size, x.shape[1], 
                                    tile_lenght - buffer,
                                    tile_lenght - buffer)
                else:
                    tile_ij = x[:, :,
                                i * tile_length: (i + 1) * tile_length,
                                j * tile_length: (j + 1) * tile_length]
        
                for k in range(batch_size):
                    num_channels = tile_ij.shape[1]

                    # leave a random gap between tiles to avoid shortcuts due to edge continuity
                    if self.buffer:

                        buffer_x1, buffer_x2 = np.random.multinomial(buffer, [0.5, 0.5])
                        buffer_y1, buffer_y2 = np.random.multinomial(buffer, [0.5, 0.5])

                        tile_x1 = i * tile_lenght + buffer_x1 
                        tile_y1 = (i + 1) * tile_lenght - buffer_x2
                        tile_x2 = j * tile_lenght + buffer_y1 
                        tile_y2 = (j + 1) * tile_lenght - buffer_y2

                        tile_ij[k] = x[k, :, tile_x1: tile_x2, tile_y1: tile_y2]

                    # random spacial jitter goes here

                    # nomralization goes here

                tiles.append(tile_ij)
        
        # tensorizing tiles
        tiles = torch.stack(tiles)

        # randomly shuffle tiles
        y = []
        for i in range(batch_size):
            permutation_index = np.random.randint(0, self.num_permutations)
            permutation = torch.tensor(self.permutations[permutation_index])

            tiles[:, i, :, :] = tiles[permutation, i, :, :]
            y.append(permutation_index)
        
        y = torch.tensor(y).long()

        return tiles, y 
    
    @staticmethod
    def generate_permutation_set(num_tiles, num_permutations, method="maximal"):
        """
        Details
        """ 

        if method not in ["maximal", "average", "minimal"]:
            raise ValueError("The specific method=%s is not recoginised!" % method)

        pemutations = []

        # get all permutations
        tile_positions = list(range(num_tiles))
        all_permutations = list(itertools.permutations(tile_positions))

        # convert all permutations to 2D matrix
        all_permutations = np.array(all_permutations).T

        # uniformly sample out of (num_tiles) indeces to initialise
        current_index = random.randint(0, np.math.factorial(num_tiles) - 1)

        for i in tqmd(range(1, num_permutations +1), desc = "Generating Permutation Set"):
            # adding permutations at current index to set
            permutations.append(tuple(all_permutations[:, current_index]))
            # remove current permutations at current index from all permutations
            all_permutations = np.delete(all_permutations, current_index, axis=1)

            # uniformly sample if average and skip computation
            if method == "average":
                current_index = random.randint(0, np.math.factorial(num_tiles) - i)
                continue
            
            # compute the hamming distance matrix
            distances = np.empty((i, np.math.factorial(num_tiles) - i))

            for j in range(i):
                for k in range(np.math.factorial(num_tiles) - i):
                    distances[j, k] = hamming(permutation[j], all_permutations[:, k])
            
            distances = np.matmul(np.ones((1, 1)), distances)

            # choose the next permutation s.t. it maximises objective
            if method == "maximal":
                current_index = np.argmax(distances) 
            elif method == "minimal":
                current_index = np.argmin(distances)
        
        # compute minimum hamming distance in generated permutation sets
        distances_ = []
        for i in range(num_permutations):
            for j in range(num_permutations):
                if i != j:
                    distances_.append(hamming(np.array(permutations[i]), np.array(permutations[j])))
        
        min_distance = min(distances_)
        print('Minimum hamming distance is chosen as %0.4f' % min_distance)

        return permutations