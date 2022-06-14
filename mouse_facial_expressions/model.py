import torchvision
import torch
import pandas as pd

class PretrainedResnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Transfer learning using a pretrained resnet (basically copied from pytorch docs)
        base_model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = base_model.fc.in_features
        base_model.fc = torch.nn.Linear(num_ftrs, 2)
        self.base_model = base_model
        
    def __call__(self, x):
        return self.base_model(x)
    
    
