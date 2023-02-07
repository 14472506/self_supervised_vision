# imports for data loading
import torch
from data_handler import DataHandler
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 

# loading code
mock_cd = {
    "model": {
        "name": "JigRot",
        "backbone": "pre_trained",
        "load_model": "outputs/RotNet_base/best_model.pth",
        "num_tiles": 9,
        "permutations": 100,
        "rotations": 4,
        "tile_rotations": 4
        },
    "data": {
        "path": "data/jersey_royals_ssl_ds",
        "tt_split": 0.8,
        "tv_split": 0.8,
        "train_loader": {
            "batch_size": 4,
            "shuffle": True,
            "num_workers": 8
        },
        "test_loader": {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 8
        },
        "validation_loader": {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 8
            }
        },
    }

h = DataHandler(mock_cd)
train, test, val = h.data_laoders()

# data loader test
for i, data in enumerate(train, 0):
    sample, jig_gt, rot_gt = data
    # droppign batch size
    sample = sample[0]
    break

fig, ax  =  plt.subplots(3, 3, figsize=(15, 12))

print(jig_gt.size())
print(rot_gt.size())

fig_c = 0
for i in range(3):
    for j in range(3):
        ax[i][j].imshow(sample[fig_c].permute(1, 2, 0))
        fig_c += 1
plt.show()

