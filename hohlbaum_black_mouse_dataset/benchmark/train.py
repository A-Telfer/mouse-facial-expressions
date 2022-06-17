import torch
import pandas as pd
import numpy as np
import logging
import os
import argparse
import json

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

assert torch.cuda.is_available()

print("""
  ____  __  __      __   ____                  _                          _    
 |  _ \|  \/  |    /_ | |  _ \                | |                        | |   
 | |_) | \  / |_   _| | | |_) | ___ _ __   ___| |__  _ __ ___   __ _ _ __| | __
 |  _ <| |\/| \ \ / / | |  _ < / _ \ '_ \ / __| '_ \| '_ ` _ \ / _` | '__| |/ /
 | |_) | |  | |\ V /| | | |_) |  __/ | | | (__| | | | | | | | | (_| | |  |   < 
 |____/|_|  |_| \_/ |_| |____/ \___|_| |_|\___|_| |_|_| |_| |_|\__,_|_|  |_|\_\
                                                                               
                                                                               
Unofficial PyTorch benchmark by Andre Telfer (andretelfer@cmail.carleton.ca)

Model Paper: Andresen, N., Wöllhaf, M., Hohlbaum, K., Lewejohann, L., Hellwich, O., Thöne-Reineke, C., & Belik, V. (2020). Towards a fully automated surveillance of well-being status in laboratory mice using deep learning: Starting with facial expression analysis. PLoS One, 15(4), e0228059.

Dataset: Hohlbaum, K., Andresen, N., Wöllhaf, M., Lewejohann, L., Hellwich, O., Thöne-Reineke, C., & Belik, V. (2019). Black Mice Dataset v1.

""")

###################
#      ARGS       #
###################
DEVICE = 'cuda'

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--logdir', default="logs", help='path for logfile')
parser.add_argument('--data-path', default=os.environ.get("BMv1_DATASET"), help='Path to BMv1 dataset')
parser.add_argument('--batch-size', default=100, type=int, help='Training/validation batch size')
parser.add_argument('--epochs', default=50, type=int, help='Number of epochs to train for')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers for loading dataset')
parser.add_argument('--train-ratio', default=0.9, type=float, help='Ratio of dataset used to create the train split')
parser.add_argument('--shuffles', default=1, type=int, help='Number of training shuffles to perform')

# parser.add_argument('--rotation-augmentation', action='store_true', help='Use a rotation augmentation')
args = parser.parse_args()

data_path = Path(args.data_path)
assert data_path.exists()

# Create a run subdirectory of the logdir
logdir = Path(args.logdir)
run = 0
while (logdir / f'run{run}').exists():
    run += 1
    
logdir = logdir / f'run{run}'
if not logdir.exists():
    logdir.mkdir(parents=True)

for shuffle in range(args.shuffles):
    ###################
    #     LOGGING     #
    ###################
    logger = logging.getLogger('BMv1 Benchmark')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(logdir / f"train-shuffle{shuffle}.log")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - shuffle {shuffle} - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Setup logger")

    # Show args
    logger.info(f"Arguments:\n{json.dumps(vars(args), indent=4)}")

    ###################
    #     DATASET     #
    ###################
    logger.info("Loading dataset")
    from src.dataset import BMv1
    dataset = BMv1(data_path)
    train_sampler, val_sampler = dataset.train_test_split(args.train_ratio)
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        num_workers=args.num_workers
    )
    logger.info("Loaded dataset")

    ###################
    #      Model      #
    ###################
    logger.info("Loading model")
    from src.model import PretrainedResnet
    model = PretrainedResnet()
    logger.info("Loaded model")

    ###################
    #    TRAIN LOOP   #
    ###################
    logger.info("Beginning training loop")
    ch.setLevel(logging.ERROR) # inteferes with tqdm
    
    optimizer = model.configure_optimizer()
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(DEVICE)

    def get_stats(pred, label):
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        label = label.cpu().numpy()
        accuracy = np.mean((pred == label))
        accuracy = round(accuracy, 2)
        TP = np.sum((pred==label)[label==1])
        FP = np.sum((pred!=label)[label==0])
        TN = np.sum((pred==label)[label==0])
        FN = np.sum((pred!=label)[label==1])
        return {'accuracy': accuracy, 'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
        
    for epoch in range(args.epochs):
        print("Epoch:", epoch)
        
        # Perform a pass over the training dataset
        model.train()
        optimizer.zero_grad()
        
        train_loop = tqdm(train_loader, leave=False)
        train_history = []
        for batch_index, batch in enumerate(train_loop):
            optimizer.zero_grad()
            image = batch['image'].to(DEVICE)
            label = batch['pain'].to(DEVICE)
            
            # Loss/backprop
            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()
            
            # Stats
            lr = optimizer.param_groups[0]['lr']
            history_item = {**get_stats(pred, label), 'loss': loss.item()}
            batch_summary = "Loss: {loss:02.02f}, Acc: {accuracy:01.02f}, TP {TP:02}, FP {FP:02}, TN {TN:02}, FN {FN:02}".format(**history_item)
            train_loop.desc = batch_summary
            logger.info(f"Epoch: {epoch}, Batch: {batch_index}, LR: {lr}, " + batch_summary)
            
            train_history.append(history_item)
            optimizer.step()
        
        # Summarize train epoch
        train_history = pd.DataFrame(train_history)
        train_history_agg = train_history.agg({'accuracy': 'mean', 'loss': 'mean', 'TP': 'sum', 'FP': 'sum', 'TN': 'sum', 'FN': 'sum'})
        summary = "Train averages -> Loss: {loss:02.02f}, Acc: {accuracy:01.02}, TP {TP:02.0f}, FP {FP:02.0f}, TN {TN:02.0f}, FN {FN:02.0f}"
        summary = summary.format(**train_history_agg.to_dict())
        logger.info(summary)
        print(summary)
        
        # Validation loop
        model.eval()
        val_loop = tqdm(val_loader, leave=False, desc="Validation")
        val_history = []
        for batch_index, batch in enumerate(val_loop):
            image = batch['image'].to(DEVICE)
            label = batch['pain'].to(DEVICE)
            pred = model(image)
            val_history.append(get_stats(pred, label))
        
        val_history = pd.DataFrame(val_history)
        val_history_agg = val_history.agg({'accuracy': 'mean', 'TP': 'sum', 'FP': 'sum', 'TN': 'sum', 'FN': 'sum'})
        summary = "Validation averages -> Acc: {accuracy:01.02}, TP {TP:02.0f}, FP {FP:02.0f}, TN {TN:02.0f}, FN {FN:02.0f}"
        summary = summary.format(**val_history_agg.to_dict())
        logger.info(summary)
        print(summary)