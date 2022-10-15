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
from tqdm import tqdm
import random
from scipy.spatial.distance import hamming


#import torchnet as tnt
import torch
import torch.utils.data as data
import torchvision.transforms as T
import torch.nn.functional as F

# supporting files
from permutations import hun_perm, ten_perm, perm_24

# =============================================================================================== #
# Classes
# =============================================================================================== #
# ===== Rotation Dataset Classes ================================================================ #
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

class RotationTrainAugMapper(torch.utils.data.Dataset):
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

# ===== Jigsaw Dataset Classes ================================================================== #
# =============================================================================================== #
class JigsawDataset(data.Dataset):
    """
    Detials
    """

    def __init__(self, root, num_tiles=9, num_permutations=1000, permgen_method='maximal',
                 grayscale_probability=0.3, buffer=True, jitter=True, normalization=True, seed=42):
        """
        Detials
        """
        # setting random seed
        self.seed = seed
        self.set_seed()

        # image file path and image list
        self.root = os.path.expanduser(root)
        self.image_files = []
        for file in os.listdir(self.root):
            self.image_files.append(file)

        # number of tiles and permutations
        self.num_tiles = num_tiles
        self.num_permutations = num_permutations
        self.perm_method = permgen_method
        
        # buffer for image processing
        self.buffer = buffer

        # image modification
        self.grayscale_probability = grayscale_probability
        self.jitter = jitter
        self.normalize = normalization

        #self.permutations = self.generate_permutation_set(num_tiles = self.num_tiles,
        #                                                  num_permutations = self.num_permutations,
        #                                                  method = self.perm_method)
        
        self.permutations = perm_24
        print(self.permutations)

    def __getitem__(self, idx):
        """
        Detials
        """
        # load images from index
        image_path =  os.path.join(self.root, self.image_files[idx])
        img = Image.open(image_path).convert("RGB")
        
        # process image
        img, img_width, img_height = self.process_image(img)

        # Augmentations
        transform = T.Compose([T.ToTensor()])
        torch_img = transform(img)

        img =  torch_img
    
        # gray scale goes here if gray scale is applied
        # GRAY SCALE

        # compute tile lengths and exract tiles
        tiles = []
        num_tiles_per_dimension = int(np.sqrt(self.num_tiles))
        tile_length = img_width // num_tiles_per_dimension 
 
        buffer = int(tile_length * 0.1)
        
        # entering set of loops to go through both dimensions of image
        for i in range(num_tiles_per_dimension):
            for j in range(num_tiles_per_dimension):
    
                if self.buffer:
                    tile_ij = torch.empty(img.shape[0], 
                                    tile_length - buffer,
                                    tile_length - buffer)
                    buffer_x1, buffer_x2 = np.random.multinomial(buffer, [0.5, 0.5])
                    buffer_y1, buffer_y2 = np.random.multinomial(buffer, [0.5, 0.5])
                    tile_x1 = i * tile_length + buffer_x1 
                    tile_x2 = (i + 1) * tile_length - buffer_x2
                    tile_y1 = j * tile_length + buffer_y1 
                    tile_y2 = (j + 1) * tile_length - buffer_y2
                    tile_ij = img[:, tile_x1: tile_x2, tile_y1: tile_y2]
                else:
                    tile_ij = img[:,
                                i * tile_length: (i + 1) * tile_length,
                                j * tile_length: (j + 1) * tile_length]

                # random spacial jitter goes here
                # nomralization goes here

                tiles.append(tile_ij)

        # Tensorising tiles
        tiles = torch.stack(tiles)
               
        # randomly shuffle tiles
        y = []
        
        permutation_index = np.random.randint(0, self.num_permutations)
        permutation = torch.tensor(self.permutations[permutation_index])
        tiles[:, :, :, :] = tiles[permutation, :, :, :]

        y.append(permutation_index)

        label = torch.zeros(self.num_permutations)
        label[y] = 1

        return tiles, label 
        
    def __len__(self):
        """
        Details
        """
        return len(self.image_files)

    def process_image(self, img):
        """
        Detials
        """
        # chack size and resize if needs be 
        img_width, img_height = img.size

        # if sizes do not match
        if img_width != img_height:

            # find max and min dimension    
            wh_idx = [img_width, img_height]
            min_dim = min(wh_idx)
            max_dim = max(wh_idx)

            # get crop dims 
            delta = max_dim - min_dim
            min_crop = delta/2
            max_crop = max_dim - min_crop

            # get min side index
            min_idx = wh_idx.index(min_dim)

            # configuring crop window depending on min dim index
            if min_idx == 0: 
                left = 0
                top = min_crop
                right = min_dim 
                bottom = max_crop
            elif min_idx == 1:
                left = min_crop
                top = 0
                right = max_crop
                bottom = min_dim

        # cropping image to square
        img = img.crop((left, top, right, bottom))

        img = img.resize((1080, 1080))
        
        img_width, img_height = img.size

        return img, img_width, img_height
  
    @staticmethod
    def generate_permutation_set(num_tiles, num_permutations, method="maximal"):
        """
        Details
        """ 
        if method not in ["maximal", "average", "minimal"]:
            raise ValueError("The specific method=%s is not recoginised!" % method)

        permutations = []

        # get all permutations
        tile_positions = list(range(num_tiles))
        all_permutations = list(itertools.permutations(tile_positions))

        # convert all permutations to 2D matrix
        all_permutations = np.array(all_permutations).T

        # uniformly sample out of (num_tiles) indeces to initialise
        current_index = random.randint(0, np.math.factorial(num_tiles) - 1)

        for i in tqdm(range(1, num_permutations + 1), desc='Generating Permutation Set'):
            # adding permutations at current index to set
            permutations.append(tuple(all_permutations[:, current_index]))
            # remove current permutations at current index from all permutations
        #################################################################################
        #    all_permutations = np.delete(all_permutations, current_index, axis=1)
        #
        #    # uniformly sample if average and skip computation
        #    if method == "average":
        #        current_index = random.randint(0, np.math.factorial(num_tiles) - i)
        #        continue
        #    
        #    # compute the hamming distance matrix
        #    distances = np.empty((i, np.math.factorial(num_tiles) - i))
        #
        #    for j in range(i):
        #        for k in range(np.math.factorial(num_tiles) - i):
        #            distances[j, k] = hamming(permutations[j], all_permutations[:, k])
        #    
        #    distances = np.matmul(np.ones((1, i)), distances)
        #
        #    # choose the next permutation s.t. it maximises objective
        #    if method == "maximal":
        #        current_index = np.argmax(distances) 
        #    elif method == "minimal":
        #        current_index = np.argmin(distances)
        #
        ## compute minimum hamming distance in generated permutation sets
        #distances_ = []
        #for i in range(num_permutations):
        #    for j in range(num_permutations):
        #        if i != j:
        #            distances_.append(hamming(np.array(permutations[i]), np.array(permutations[j])))
        #
        #min_distance = min(distances_)
        #print('Minimum hamming distance is chosen as %0.4f' % min_distance)
        #
        #print(permutations)
        #return permutations
    
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

class JigsawTrainAugMapper(torch.utils.data.Dataset):
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
            transformed = self.transforms(image=np_img)
            stack_image = transformed['image']
            aug_stack.append(stack_image)

        stack = torch.stack(aug_stack)
        image = stack

        return(image, label)

    def __len__(self):
        """
        Details
        """
        return len(self.dataset)


