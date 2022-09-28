
from datasets import JigsawDataset
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

root = "data/jersey_royals_ssl_ds"
dataset = JigsawDataset(root=root,
                        num_tiles=9,
                        num_permutations=10,
                        permgen_method='maximal',
                        grayscale_probability=0.3,
                        buffer=True,
                        jitter=True,
                        normalization=True)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size = 1,
                                          shuffle = True,
                                          num_workers = 8)
tiles = dataset[7]


def tensor2PIL(torch_img):
    return transforms.ToPILImage()(torch_img.squeeze(0))

fig, ax = plt.subplots(3, 3, figsize=(15, 12))
for i in range(3):
    for j in range(3):
        ax[i][j].imshow(tensor2PIL(tiles[i * 3 + j]))
plt.show()

"""
import os

from PIL import Image
import matplotlib.pyplot as plt

import torch

import torchvision.transforms as transforms

from collators import RotationCollator

transform = transforms.Compose([transforms.ToTensor()])
def tensor2PIL(torch_img):
    return transforms.ToPILImage()(torch_img.squeeze(0))

img = Image.open(os.path.join("data/jersey_royals_ssl_ds", 'GD1_001.jpg')).convert('RGB')
torch_img = transform(img).unsqueeze(0)
plt.imshow(tensor2PIL(torch_img))

#fig, ax = plt.subplots(1, 4, figsize=(15, 12))
collator = RotationCollator(num_rotations=4, rotation_procedure='all')
x, y = collator([[torch_img.squeeze(0), -1]])
#for i in range(x.shape[0]):
#    ax[i].set_title(r'$y=%d$' % y[i])
#    ax[i].imshow(tensor2PIL(x[i]))
#plt.show()
"""