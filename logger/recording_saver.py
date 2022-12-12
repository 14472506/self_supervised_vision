"""
Detials
"""
# imports
import json

# functions
def save_records(rec_dict, conf_dict):
    """
    Detials
    """
    # saving conf_dict
    with open("outputs/" + conf_dict["logging"]["experiment_name"] + "/train_data.json", "w") as f:
        json.dump(rec_dict, f)
