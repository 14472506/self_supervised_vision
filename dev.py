"""
Details
"""
# imports
# --- general
import os
from PIL import Image
import matplotlib.pyplot as plt
# --- torch
import torch
import torchvision.transforms as transforms
# --- package_files
from collators import RotationCollator

# --- functions
transform = transforms.Compose([transforms.ToTensor()])

def tensor2PIL(torch_img):
    return transforms.ToPILImage()(torch_img.squeeze(0))

# dev 
# --- loading image
path_to_image = "jersey_royals/08.JPG"
img = Image.open(path_to_image).convert('RGB')
#plt.imshow(img)

# --- getting torch image
torch_img = transform(img).unsqueeze(0)
#plt.imshow(tensor2PIL(torch_img))
#plt.show()

# --- rotation collator
fig, ax = plt.subplots(1, 4, figsize=(20, 20))

collator = RotationCollator(num_rotations=4, rotation_procedure='random')
x, y = collator([[torch_img.squeeze(0), -1]])

print(x)
print(y)

#for i in range(x.shape[0]):
#    ax[i].set_title('r$y=%d$' % y[i])
#    ax[i].imshow(tensor2PIL(x[i]))
#
#plt.show()