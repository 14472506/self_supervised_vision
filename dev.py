from datasets import JigsawDataset
import torch

root = "data/jersey_royals_ssl_ds"
dataset = JigsawDataset(root=root,
                        num_tiles=9,
                        num_permutations=1000,
                        permgen_method='maximal',
                        grayscale_probability=0.3,
                        buffer=True,
                        jitter=True,
                        normalization=True)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size = 1,
                                          shuffle = True,
                                          num_workers = 8)

dataset[0]