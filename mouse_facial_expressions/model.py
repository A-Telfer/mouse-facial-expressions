import torchvision
import torch
import pandas as pd

class PretrainedResnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Transfer learning using a pretrained resnet (basically copied from pytorch docs)
        base_model = torchvision.models.resnet50(pretrained=True)
    
        # Freeze the parameters of the original model
        for param in base_model.parameters():
            param.requires_grad = False
            
        # TODO: check how keras removes the "head" (cutoff around 1024 features)
        # Create a new output layer with unfrozen parameters
        num_ftrs = base_model.fc.in_features 
        base_model.fc = torch.nn.Linear(num_ftrs, 2)
        self.base_model = base_model
        
    def __call__(self, x):
        return self.base_model(x)
    
    def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.base_model.fc.parameters(), lr=0.001, betas=[0.9, 0.999], eps=1e-7, weight_decay=0)
        # optimizer = torch.optim.SGD(self.base_model.fc.parameters(), lr=0.001, momentum=0.9)
        return optimizer
    
    
