import torch

x = torch.rand(9, 512)
x = x.view(x.shape[1], -1)

print(x.shape)