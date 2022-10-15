"""
Detials
"""
# imports
import torch
from models import JigsawClassifier
from utils import model_saver

class Exporter():
    """
    Details 
    """

    def __init__(self, model_path):
        """
        Detials
        """
        self.model_path = model_path
        self.model = JigsawClassifier(pre_trained=True)
        self.save_name = "jigsaw_backbone.pth"
        self.load_model()
        self.saver()
        

    def load_model(self):
        """
        Detials
        """
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["state_dict"])
    
    def saver(self):
        
        model = self.model.backbone
        checkpoint = {"state_dict": model.state_dict()}

        # saving last model
        torch.save(checkpoint, self.save_name)

# execution 
if __name__ == "__main__":

    # model path
    model_path = "outputs/Jigsaw_A5e-5_bs4_40ep/best_model.pth"
    Exporter(model_path)