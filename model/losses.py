"""
Detials
"""
# imports
import numpy as np
import torch
import torch.nn.functional as F

# loss functions
def classification_loss(y_hat, y):
    """
    Detials
    """
    loss = F.cross_entropy(y_hat, y)
    return loss