"""
Details
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

# =============================================================================================== #
# Models
# =============================================================================================== #
def resnet50_rotation_classifier(pre_trained=True, num_rotations=4):
    """
    Details
    """
    # load model
    if pre_trained:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = resnet50(weights=None)
    
    # get number of feat from model 
    num_features = model.fc.in_features

    # make adjust final layer to fit number of rotations
    model.fc = nn.Linear(num_features, num_rotations)

    # return model
    return model