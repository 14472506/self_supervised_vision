"""
Detials
"""
# imports
import json
import os
import errno

# fucntions
def save_config(conf_dict):
    """
    Details
    """
    # make directory
    try:
        os.makedirs("outputs/" + conf_dict["logging"]["experiment_name"])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # saving conf_dict
    with open("outputs/" + conf_dict["logging"]["experiment_name"] + "/exp_config.json", "w") as f:
        json.dump(conf_dict, f)