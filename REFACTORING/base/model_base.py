"""
name        : model_base.py

task        : provides base structure for models to inherit

edited by   : bradley hurst
"""
# imports
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

# class
class ModelBase(nn.Module):
    """
    Details
    """
    def __init__(self, pre_trained=True):
        """
        Detials
        """
        super(ModelBase, self).__init__()
        self.pre_trained = pre_trained
        self.backbone = self.backbone_selector(self.pre_trained)


    def backbone_selector(self, pre_trained):
        """
        Detials
        """
        # either load pre-trained weights or dont
        if pre_trained:
            backbone = resnet50(weights=ResNet50_Weights)
        else:
            backbone = resnet50(weigths=None)
        
        # return backbone
        return backbone