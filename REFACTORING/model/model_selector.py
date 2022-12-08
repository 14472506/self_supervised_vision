"""
Detials
"""
# imports
from .models import RotNet, Jigsaw
from .optimisers import OptimSelector
from .losses import classification_loss

# selector functions
def rotnet_setup():
    """
    Detials
    """
    
    model = RotNet()
    optimiser = OptimSelector(model.parameters(), "Adam").selector()
    criterion = classification_loss

    return model, optimiser, criterion

def jigsaw_setup(num_tiles=9, num_permutations=100):
    """
    Detials
    """

    model = Jigsaw(num_tiles=num_tiles, num_permutations=num_permutations)
    optimiser = OptimSelector(model.parameters(), "Adam").selector()
    criterion = classification_loss

    return model, optimiser, criterion