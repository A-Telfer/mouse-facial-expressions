from pickletools import optimize
import torch
import torchvision
import pandas as pd
import numpy as np
import logging

from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imread
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.utils import make_grid

assert torch.cuda.is_available()

EPOCHS = 50
BATCH_SIZE = 256
DEVICE = 'cuda'

###################
#     LOGGING     #
###################
logger = logging.getLogger('mouse_facial_expressions')
logger.setLevel(logging.INFO)

fh = logging.FileHandler('model3.log')
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

###################
#     DATASET     #
###################
from mouse_facial_expressions.dataset import BMv1
dataset = BMv1("/home/andre/shared/curated/BMv1/")
train_sampler, val_sampler = dataset.train_test_split()
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

###################
#      Model      #
###################
from mouse_facial_expressions.model import PretrainedResnet
model = PretrainedResnet()

###################
#    TRAIN LOOP   #
###################
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
optimizer = model.configure_optimizer()

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()
model = model.to(DEVICE)
for epoch in range(EPOCHS):
    print("Epoch:", epoch)
    
    # Perform a pass over the training dataset
    model.train()
    optimizer.zero_grad()
    
    loop = tqdm(train_loader)
    for batch_index, batch in enumerate(loop):
        image = batch['image'].to(DEVICE)
        label = batch['pain'].to(DEVICE)
        
        # Loss/backprop
        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        
        accuracy = np.mean((torch.argmax(pred[0]) == label).cpu().numpy())
        accuracy = round(accuracy, 2)
        loop.desc = f"Loss: {round(loss.item(),2)}, Acc: {accuracy}"
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch: {epoch}, Batch: {batch_index}, LR: {lr}, Loss: {round(loss.item(),2)}, Acc: {accuracy}")
    
    # scheduler.step()
            
    # TODO
    model.eval()
    
    for idx in val_sampler:
        pass