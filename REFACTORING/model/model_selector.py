"""
Detials
"""
# imports
from .models import RotNet
from .optimisers import OptimSelector
from .losses import classification_loss

def rotnet_setup():
    """
    Detials
    """
    
    model = RotNet()
    optimiser = OptimSelector(model.parameters(), "Adam").selector()
    criterion = classification_loss

    return model, optimiser, criterion