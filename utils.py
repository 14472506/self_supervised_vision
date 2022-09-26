"""
Detials
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
import os
import errno
import torch

# =============================================================================================== #
# Functions
# =============================================================================================== #
def make_dir(path):
    """
    Detials
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def model_saver(epoch, model, optimizer, path, pth_name):
    """
    Detials
    """
    # check if output dir exists, if not make dir
    make_dir(path)

    # making save dictionary for model.pth
    checkpoint = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    # saving last model
    last_model_path = path + "/" + pth_name
    torch.save(checkpoint, last_model_path)