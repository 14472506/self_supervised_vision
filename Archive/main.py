"""
Detials
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
from loops import Training_loop
import yaml

# =============================================================================================== #
# Main
# =============================================================================================== #
def main():
    # defining list of experiments    
    exp_list = ["experiment_configs/test.yaml"]

    # looping through experiments list calling loop_train
    for exp in exp_list:

        print("running experiment: ", exp)

        with open(exp, "r") as data:
            try: 
                config_dict = yaml.safe_load(data)
            except yaml.YAMLError as exc:
                print(exc)
    
        Training_loop(cd = config_dict) 

# =============================================================================================== #
# Execution
# =============================================================================================== #
if __name__ == "__main__":
    main()   