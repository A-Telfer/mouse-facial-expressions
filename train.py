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
BATCH_SIZE = 100
DEVICE = 'cuda'
DATAPATH = "/home/andretelfer/Downloads/BMv1/"
# DATAPATH = "/home/andretelfer/shared/curated/BMv1/"

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
dataset = BMv1(DATAPATH)
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
# criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.BCELoss()
criterion = torch.nn.CrossEntropyLoss()

model = model.to(DEVICE)
for epoch in range(EPOCHS):
    print("Epoch:", epoch)
    
    # Perform a pass over the training dataset
    model.train()
    optimizer.zero_grad()
    
    loop = tqdm(train_loader)
    for batch_index, batch in enumerate(loop):
        optimizer.zero_grad()
        image = batch['image'].to(DEVICE)
        label = batch['pain']
        
        # label = torch.nn.functional.one_hot(label, 2).float()
        label = label.to(DEVICE)
        
        # Loss/backprop
        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        label = label.cpu().numpy()
        # label = torch.argmax(label, dim=1).cpu().numpy()
        accuracy = np.mean((pred == label))
        accuracy = round(accuracy, 2)
        TP = np.sum((pred==label)[label==1])
        FP = np.sum((pred!=label)[label==0])
        TN = np.sum((pred==label)[label==0])
        FN = np.sum((pred!=label)[label==1])
        
        loop.desc = f"Loss: {round(loss.item(),2)}, Acc: {accuracy}, TP {TP}, FP {FP}, TN {TN}, FN {FN}"
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch: {epoch}, Batch: {batch_index}, LR: {lr}, Loss: {round(loss.item(),2)}, Acc: {accuracy}, TP {TP}, FP {FP}, TN {TN}, FN {FN}")
        optimizer.step()
        
    # scheduler.step()
            
    # TODO
    model.eval()
    
    for idx in val_sampler:
        pass