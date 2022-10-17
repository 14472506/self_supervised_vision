"""
Detials - something along the lines of these are the losses for the methods
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
import numpy as np
import torch
import torch.nn.functional as F

# =============================================================================================== #
# Classes
# =============================================================================================== #
def classification_loss(y_hat, y):
    """
    Detials and uses
    """
    loss = F.cross_entropy(y_hat, y)
    return loss

def reconstruction_loss(y_hat, y, mask=None, reduction='mean'):
    """
    Detials
    """
    loss = F.mse_loss(y_hat, y, reduction=reduction)
    if mask is not None:
        loss *= mask
    return loss