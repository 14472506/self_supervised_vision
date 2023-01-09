"""
Details
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

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

        self.twin_network = nn.Sequential(nn.Linear(1000, 512, bias=False),
                                          nn.BatchNorm1d(512),
                                          nn.ReLU(inplace=True))
        
        self.classifier = nn.Sequential(nn.Linear(4608, 4096, bias=False),
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
        print(x.shape)
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

class ContextEncoder(nn.Module):
    """
    Details
    """
    def __init__(self, target_size, dropout_rate=0.5):
        super(ContextEncoder, self).__init__()
        self.latent_dim = 2048 * 1 * 1

        self.backbone = self.backbone_selector()
        self.encoder = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
        #self.encoder[8] = nn.AdaptiveAvgPool2d((1, 1))
     
        self.bottleneck = nn.Sequential(nn.Conv1d(in_channels=self.latent_dim,
                                                   out_channels=self.latent_dim,
                                                   kernel_size=1,
                                                   groups=self.latent_dim),
                                         nn.Dropout() if dropout_rate > 0. else nn.Identity())

        self.decoder = self.decoder_head(target_size=target_size, batch_norm=True)

    def forward(self, x):
        # resnet encoder output
        x = self.encoder(x)
        x = torch.flatten(x, 2)      
        x = self.bottleneck(x)
        x = x.view(-1, 2048, 1, 1)         
        x = self.decoder(x)

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

    def decoder_head(self, target_size, batch_norm=True):
        """
        Detials
        """
        # define decoder
        decoder = nn.Sequential(nn.ConvTranspose2d(2048, 128, kernel_size=4, stride=2),
                           nn.BatchNorm2d(128) if batch_norm else nn.Identity(),
                           nn.ReLU(inplace=True),
                           nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
                           nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
                           nn.ReLU(inplace=True),
                           nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2),
                           nn.BatchNorm2d(64) if batch_norm else nn.Identity(),
                           nn.ReLU(inplace=True),
                           nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
                           nn.BatchNorm2d(32) if batch_norm else nn.Identity(),
                           nn.ReLU(inplace=True),
                           nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2),
                           nn.BatchNorm2d(3) if batch_norm else nn.Identity(),
                           nn.ReLU(inplace=True),
                           nn.Upsample(target_size, mode='bilinear', align_corners=True))

        # somethings happening here
        decoder = nn.Sequential(*[child for child in decoder.children() if not isinstance(child, nn.Identity)])

        return decoder
            