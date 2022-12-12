"""
Detials
"""
# imports
import torch.optim as optim

# classes
class OptimSelector():
    """
    Detials
    """
    def __init__(self, model_params, conf_dict):
        """
        Details
        """
        self.model_params = model_params
        self.cd = conf_dict

    def selector(self):
        """ 
        Detials
        """
        if self.cd["optimiser"]["name"] == "Adam":
            optimizer = optim.Adam(self.model_params,
                            lr = self.cd["optimiser"]["lr"])
        elif self.optim_flag == "SGD":
            optimizer = optim.SGD(self.model_params,
                            lr = self.cd["optimiser"]["lr"],
                            momentum = self.cd["optimiser"]["momentum"],
                            weight_decay = self.cd["optimiser"]["weight_decay"])
        else:
            print("Undefined Optimiser")

        # return selected optimiser
        return optimizer