import torch
import torchvision

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
    