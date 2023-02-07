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
    def __init__(self, model, conf_dict):
        """
        Details
        """
        self.model = model
        self.cd = conf_dict

    def selector(self):
        """ 
        Detials
        """
        if self.cd["optimiser"]["name"] == "Adam":
            optimizer = optim.Adam([{"params": self.model.backbone.parameters()},
                                    {"params": self.model.rot_classifier.parameters()},
                                    {"params": self.model.twin_network.parameters()},
                                    {"params": self.model.classifier.parameters()}],
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