import torch
import torch.nn.functional as F

#loss = F.cross_entropy

inputs = torch.randn(24, 5, requires_grad=True)
target = torch.empty(24, dtype=torch.long).random_(5)

loss = F.cross_entropy(inputs, target) 

print(loss)