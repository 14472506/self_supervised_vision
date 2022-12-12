"""
Detials
"""
# functions
def recorder_dict():
    rec_dict = {
        "epoch": [],
        "training_loss": [],
        "validation_loss": [],
        "best_epoch": [],
        "best_training_loss": [],
        "best_validation_loss": []
    }
    return rec_dict