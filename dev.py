# loadinging and nomalizing dataset
import torch
import torchvision
import torchvision.transforms as transforms 

from datasets import RotationDataset

transfrom = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
)

batch_size = 1

trainset = RotationDataset(root = "./jersey_royals")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
                                        
#import matplotlib.pyplot as plt

#import numpy as np 
#
#def imshow(img):
#    img = img/2+0.5
#    npimp = img.numpy()
#    plt.imshow(np.transpose(npimp, (1 ,2 ,0)))
#    plt.show()
#
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#
#imshow(torchvision.utils.make_grid(images))
#print(" ".join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# defining model
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
model = resnet50(weights=ResNet50_Weights.DEFAULT)
num_feautes = model.fc.in_features
model.fc = nn.Linear(num_feautes, 4)

# Define loss
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

# train network
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        print(data)
    
        # get data
        input, labels = data
        
        # set param gradient to zero
        optimizer.zero_grad()
        
        # forward + backward + optimizer
        outputs = model(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # printing stats
        running_loss += loss.item()
        if i % 2000 == 1999: # printing every 200 minibatches
            print(f"[{epoch + 1}, {i +  1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0

print("finished training")