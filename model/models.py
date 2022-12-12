"""
Details
"""
# imports 
import torch.nn as nn
import torch

from base import ModelBase

# classes
class RotNet(ModelBase):
    """
    Detials
    """
    def __init__(self, conf_dict, num_rotations=4, dropout_rate=0.5, batch_norm=True):
        """
        Detials
        """
        super().__init__(conf_dict)

        self.classifier = nn.Sequential(nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                nn.Linear(1000, 4096, bias=False if batch_norm else True),
                                nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                nn.Linear(4096, 4096, bias=False if batch_norm else True),
                                nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Linear(4096, num_rotations))

                # Remove any potential nn.Identity() layers
        self.classifier = nn.Sequential(*[child for child in self.classifier.children() if not isinstance(child, nn.Identity)])

    def forward(self, x):
        """
        Detials
        """
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class Jigsaw(ModelBase):
    """
    Details
    """
    def __init__(self, conf_dict, num_tiles=9, num_permutations=10):
        """
        Detials
        """
        super().__init__(conf_dict)

        self.num_tiles = num_tiles

        self.twin_network = nn.Sequential(nn.Linear(1000, 512, bias=False),
                                          nn.BatchNorm1d(512),
                                          nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(nn.Linear(512*self.num_tiles, 4096, bias=False),
                                         nn.BatchNorm1d(4096),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4096, num_permutations))

    def forward(self, x):
        """
        Detials
        """
        assert x.shape[1] == self.num_tiles
        device = x.device
        x = torch.stack([self.twin_network(self.backbone(tile)) for tile in x]).to(device)
        x = torch.flatten(x, start_dim = 1)
        x = self.classifier(x)


        return x
    



