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
                                nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                nn.Linear(4096, 1000, bias=False if batch_norm else True),
                                nn.BatchNorm1d(1000) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Linear(1000, num_rotations))

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

class JigRot(ModelBase):
    """
    Detials
    """
    def __init__(self, conf_dict, num_tiles=9, num_permutations=100, num_rotations=4,
        dropout_rate=0.5, batch_norm=True):
        """
        Detials
        """
        # init from model base
        super().__init__(conf_dict)
        self.num_tiles = num_tiles
        self.num_rotations = num_rotations
        self.num_permutations = num_permutations

        # current ############################################################
        ## ----- rotation classifier head
        #self.rot_classifier1 = nn.Sequential(nn.Dropout() if dropout_rate > 0. else nn.Identity(),
        #                        nn.Linear(1000, 4096, bias=False if batch_norm else True),
        #                        nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
        #                        nn.ReLU(inplace=True))
        #self.rot_classifier1 = nn.Sequential(*[child for child in self.rot_classifier1.children() if not isinstance(child, nn.Identity)])

        #self.rot_classifier2 = nn.Sequential(nn.Dropout() if dropout_rate > 0. else nn.Identity(),
        #                        nn.Linear(4096*self.num_tiles, 4096, bias=False if batch_norm else True),
        #                        nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
        #                        nn.ReLU(inplace=True),
        #                        nn.Linear(4096, self.num_permutations))
        #self.rot_classifier2 = nn.Sequential(*[child for child in self.rot_classifier2.children() if not isinstance(child, nn.Identity)])
        
        ############################################################

        self.rot_classifier = nn.Sequential(nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                nn.Linear(1000, 4096, bias=False if batch_norm else True),
                                nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                nn.Linear(4096, 4096, bias=False if batch_norm else True),
                                nn.BatchNorm1d(4096) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Dropout() if dropout_rate > 0. else nn.Identity(),
                                nn.Linear(4096, 1000, bias=False if batch_norm else True),
                                nn.BatchNorm1d(1000) if batch_norm else nn.Identity(),
                                nn.ReLU(inplace=True),
                                nn.Linear(1000, num_rotations))
        self.rot_classifier = nn.Sequential(*[child for child in self.rot_classifier.children() if not isinstance(child, nn.Identity)])


        # ----- Jigsaw classifier head
        self.twin_network = nn.Sequential(nn.Linear(1000, 512, bias=False),
                                          nn.BatchNorm1d(512),
                                          nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(nn.Linear(512*self.num_tiles, 4096, bias=False),
                                         #nn.BatchNorm1d(4096),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(4096, self.num_permutations))

    def forward(self, x):
        """
        Details
        """
        assert x.shape[1] == self.num_tiles
        device = x.device

        # backbone
        x = torch.stack([self.backbone(tile) for tile in x]).to(device)
        #print("bb_out_shape: ",x.shape)

        # jigsaw
        x1 = torch.stack([self.twin_network(tile) for tile in x]).to(device)
        #print("jig_tn_out_shape: ",x1.shape)
        x1 = torch.flatten(x1, start_dim = 1)
        #print("flattened_jig_tn_out_shape: ",x1.shape)
        x1 = self.classifier(x1)
        #print("jig_class_out_shape: ",x1.shape)

        # rotnet
        x2 = torch.stack([self.rot_classifier(tile)for tile in x]).to(device)
        #print("jr_c_out_shape: ",x2.shape)
        #x2 = torch.flatten(x2, start_dim = 1)
        #print("flattened_jr_c1_out_shape: ",x1.shape)
        #x2 = self.rot_classifier2(x2)
        #print("jr_c2_out_shape: ",x1.shape)

        # returning outputs 1(jig) and 2(rot)
        return x1, x2