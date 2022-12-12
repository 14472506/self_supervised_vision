"""
Detials
"""
# imports
import os
import torch
import errno

# functions
def model_saver(path, pth_name, epoch, model, optimizer):
    """
    Detials
    """
    # make directory
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    # making save dictionary for model.pth
    checkpoint = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    # saving last model
    last_model_path = path + "/" + pth_name
    torch.save(checkpoint, last_model_path)

