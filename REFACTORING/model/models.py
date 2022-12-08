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
    def __init__(self, pre_trained=True, num_rotations=4, dropout_rate=0.5, batch_norm=True):
        """
        Detials
        """
        super().__init__(pre_trained=pre_trained)

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
    



