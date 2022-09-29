
from models import JigsawClassifier

model = JigsawClassifier()
print(model)

############################################################################################

#from datasets import JigsawDataset
#import torch
#import matplotlib.pyplot as plt
#import torchvision.transforms as transforms
#
#
#root = "data/jersey_royals_ssl_ds"
#dataset = JigsawDataset(root=root,
#                        num_tiles=9,
#                        num_permutations=10,
#                        permgen_method='maximal',
#                        grayscale_probability=0.3,
#                        buffer=True,
#                        jitter=True,
#                        normalization=True)
#data_loader = torch.utils.data.DataLoader(dataset,
#                                          batch_size = 1,
#                                          shuffle = True,
#                                          num_workers = 8)
#tiles, y = dataset[7]
#
#def tensor2PIL(torch_img):
#    return transforms.ToPILImage()(torch_img.squeeze(0))
#
#fig, ax = plt.subplots(3, 3, figsize=(15, 12))
#for i in range(3):
#    for j in range(3):
#        ax[i][j].set_title(r'$t_{%d}$' % (dataset.permutations[y[0].item()][i * 3 + j]))
#        ax[i][j].imshow(tensor2PIL(tiles[i * 3 + j]))
#
#print(y)
#plt.show()

###############################################################################################