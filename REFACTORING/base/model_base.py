"""
name        : model_base.py

task        : provides base structure for models to inherit

edited by   : bradley hurst
"""
# imports
#from saver import 
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

from saver import backbone_loader

# class
class ModelBase(nn.Module):
    """
    Details
    """
    def __init__(self, conf_dict):
        """
        Detials
        """
        super(ModelBase, self).__init__()
        self.cd = conf_dict
        self.backbone = self.backbone_selector(self.cd["model"]["backbone"])


    def backbone_selector(self, pre_trained):
        """
        Detials
        """
        # either load pre-trained weights or dont
        if pre_trained == "pre_trained":
            backbone = resnet50(weights=ResNet50_Weights)
        elif pre_trained == "load":
            backbone = backbone_loader(self.cd["model"]["load_model"])
        else:
            backbone = resnet50()
        
        # return backbone
        return backbone