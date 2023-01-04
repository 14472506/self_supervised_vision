"""
Detials
"""
# imports
from .models import RotNet, Jigsaw, JigRot
from .optimisers import OptimSelector
from .losses import classification_loss

# selector functions
def rotnet_setup(conf_dict):
    """
    Detials
    """
    model = RotNet(conf_dict=conf_dict, 
                num_rotations=conf_dict["model"]["rotations"],
                dropout_rate=conf_dict["model"]["dropout"],
                batch_norm=conf_dict["model"]["batch_norm"])
    optimiser = OptimSelector(model.parameters(), conf_dict).selector()
    criterion = classification_loss

    return model, optimiser, criterion

def jigsaw_setup(conf_dict):
    """
    Detials
    """
    model = Jigsaw(conf_dict=conf_dict,
                num_tiles=conf_dict["model"]["num_tiles"],
                num_permutations=conf_dict["model"]["permutations"])
    optimiser = OptimSelector(model.parameters(), conf_dict).selector()
    criterion = classification_loss

    return model, optimiser, criterion

def jigrot_setup(conf_dict):
    """
    Detials
    """
    model = JigRot(conf_dict=conf_dict,
                num_tiles = conf_dict["model"]["num_tiles"],
                num_permutations = conf_dict["model"]["num_perms"],
                num_rotations=conf_dict["model"]["rotations"],
                )
    optimiser = OptimSelector(model.parameters(), conf_dict).selector()
    criterion = classification_loss

    return model, optimiser, criterion
