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

class JigsawClassifier(nn.Module):
    """
    Details
    """
    def __init__(self, pre_trained=True, num_tiles=9, num_permutations=10):
        """
        Detials
        """
        super(JigsawClassifier, self).__init__()
        self.num_tiles = num_tiles
        self.backbone = self.backbone_selector()

        self.twin_network = nn.ModuleList([self.backbone,
                                          nn.Linear(1000, 512, bias=False),
                                          nn.BatchNorm1d(512),
                                          nn.ReLU(inplace=True)])
        
        self.classifier = nn.ModuleList([nn.Linear(512 * num_tiles, 4096, bias=False),
                                         nn.BatchNorm1d(4096),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4096, num_permutations)])

    def forward(self, x):
        """
        Detials
        """
        assert x.shape[0] == self.num_tiles
        device = x.device

        x = torch.stack([self.twin_network(tile) for tile in x]).to(device)
        x = x.view(x.shape[1], -1) # concatenate features from different tiles
        x = self.classifier(x)
        return x

    def backbone_selector(self, pre_trained=True):
        """
        Detials
        """
        if pre_trained:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            model = resnet50(weights=None)
        return model