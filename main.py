from tools import TrainingLoop
import torch
import json 

from saver import backbone_loader

with open("config/JigRot_config.json", "r") as f:
    conf_dict = json.load(f)

loop = TrainingLoop(conf_dict)
loop.loop()
