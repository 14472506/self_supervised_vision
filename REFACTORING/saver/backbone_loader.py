"""
Detials
"""
# imports
import torch
from torchvision.models import resnet50

# functions
def backbone_loader(model_path):
    """
    Detials
    """
    blank_backbone = resnet50()
    checkpoint = torch.load(model_path) 
    blank_backbone.load_state_dict(checkpoint["state_dict"], strict=False)
    return blank_backbone