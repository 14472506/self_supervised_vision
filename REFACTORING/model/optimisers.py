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
    def __init__(self, model_params, optim_flag, optim_conf={"lr": 0.00005}):
        """
        Details
        """
        self.optim_flag = optim_flag
        self.optim_conf = optim_conf
        self.model_params = model_params

    def selector(self):
        """ 
        Detials
        """
        if self.optim_flag == "Adam":
            optimizer = optim.Adam(self.model_params,
                            lr = self.optim_conf["lr"])
        elif self.optim_flag == "SGD":
            optimizer = optim.SGD(self.model_params,
                            lr = self.optim_conf["lr"],
                            momentum = self.optim_conf["momentum"],
                            weight_decay = self.optim_conf["weight_decay"])
        else:
            print("Undefined Optimiser")

        # return selected optimiser
        return optimizer