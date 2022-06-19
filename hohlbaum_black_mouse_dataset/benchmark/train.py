import torch
import pandas as pd
import numpy as np
import logging
import os
import argparse
import json
import matplotlib.pyplot as plt

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
RUN_DIR = Path('shared/runs')

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--save-model', action='store_true', help='Save the model at the end of training')
parser.add_argument('--save-dir', default="", help='subfolder of "runs" to save results')
parser.add_argument('--batch-size', default=os.environ.get("BATCH_SIZE", 100), type=int, help='Training/validation batch size')
parser.add_argument('--shuffles', default=os.environ.get("SHUFFLES", 1), type=int, help='Number of training shuffles to perform')
parser.add_argument('--epochs', default=os.environ.get("EPOCHS", 50), type=int, help='Number of epochs to train for')
parser.add_argument('--num-workers', default=os.environ.get("WORKERS", 4), type=int, help='Number of workers for loading dataset')
parser.add_argument('--train-ratio', default=os.environ.get("TRAIN_RATIO", 0.9), type=float, help='Ratio of dataset used to create the train split')
parser.add_argument('--data-path', default=os.environ.get("DATA_PATH"), help='Path to BMv1 dataset')
parser.add_argument('--augmentation', default=os.environ.get("AUGMENTATION", 'none'), help='Augmentation strategy (none, basic, trivial-wide)')
parser.add_argument('--optimizer', default=os.environ.get("OPTIMIZER", 'adam'), help='adam/sgd')
parser.add_argument('--learning-rate', default=os.environ.get("LEARNING_RATE", 0.001), type=float, help='adam/sgd')

# Validate arguments
args = parser.parse_args()

data_path = Path(args.data_path)
assert data_path.exists()
assert args.augmentation in ["none", "baseline", "trivial-wide", "randaug"]
assert args.optimizer in ["sgd", 'adam']

# Create a run subdirectory of the logdir
save_dir = RUN_DIR / args.save_dir
run = 0
while (save_dir / f'run{run}').exists():
    run += 1
    
run_dir = save_dir / f'run{run}'

###################
#   Per Shuffle   #
###################
for shuffle in range(args.shuffles):
    shuffle_dir = run_dir / f'shuffle{shuffle}'
    if not shuffle_dir.exists():
        shuffle_dir.mkdir(parents=True)
        
    ###################
    #     LOGGING     #
    ###################
    logger = logging.getLogger('BMv1 Benchmark')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(shuffle_dir / f"train.log")
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
    from src.dataset import BMv1, BaselineAugmentation
    from torchvision.transforms import RandAugment, TrivialAugmentWide
    from torch.utils.data import DataLoader, SubsetRandomSampler
    
    # Get the augmentation strategy
    if args.augmentation == 'none':
        augmentation = None
    elif args.augmentation == 'baseline':
        augmentation = BaselineAugmentation()
    elif args.augmentation == 'randaug':
        augmentation = RandAugment()
    elif args.augmentation == 'trivial-wide':
        augmentation = TrivialAugmentWide()
    
    # Load the dataset
    dataset = BMv1(data_path, training_transform=augmentation)
    
    # Train/test split based on animal id
    ids = dataset.labels.id.unique()
    train_size = int(len(ids) * args.train_ratio)
    train_ids = set(np.random.choice(ids, train_size, replace=False))
    train_indices = dataset.labels[dataset.labels.id.isin(train_ids)].index.tolist()
    train_sampler = SubsetRandomSampler(train_indices)

    val_ids = set(ids) - train_ids
    val_indices = dataset.labels[dataset.labels.id.isin(val_ids)].index.tolist()
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create batch loaders
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
    
    scheduler = None
    lr = args.learning_rate
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.base_model.fc.parameters(), lr=lr, betas=[0.9, 0.999], eps=1e-7, weight_decay=0)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.base_model.fc.parameters(), lr=lr, momentum=0.9) 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
        
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
        
    train_history = []
    val_history = []
    for epoch in range(args.epochs):
        print("Epoch:", epoch)
        
        # Perform a pass over the training dataset
        dataset.train = True
        model.train()
        optimizer.zero_grad()
        
        train_loop = tqdm(train_loader, leave=False)
        epoch_train_history = []
        for batch_index, batch in enumerate(train_loop):
            optimizer.zero_grad()
            image = batch['image'].float().to(DEVICE)
            label = batch['pain'].to(DEVICE)
            
            # Loss/backprop
            pred = model(image)
            loss = criterion(pred, label)
            loss.backward()
            
            # Stats
            lr = optimizer.param_groups[0]['lr']
            history_item = {**get_stats(pred, label), 'loss': loss.item(), 'lr': lr}
            batch_summary = "LR: {lr:.01e}, Loss: {loss:02.02f}, Acc: {accuracy:01.02f}, TP {TP:02}, FP {FP:02}, TN {TN:02}, FN {FN:02}".format(**history_item)
            train_loop.desc = batch_summary
            logger.info(f"Epoch: {epoch}, Batch: {batch_index}, LR: {lr}, " + batch_summary)
            
            epoch_train_history.append(history_item)
            optimizer.step()
        
        # Update the scheduler every epoch
        if scheduler is not None:
            scheduler.step()
        
        # Summarize train epoch
        epoch_train_history = pd.DataFrame(epoch_train_history)
        epoch_train_history_agg = epoch_train_history.agg({
            'accuracy': 'mean', 'loss': 'mean', 'lr': lambda x: x.iloc[0], 'TP': 'sum', 'FP': 'sum', 'TN': 'sum', 'FN': 'sum'})
        epoch_train_history_agg['epoch'] = epoch
        epoch_train_history_agg['shuffle'] = shuffle
        train_history.append(epoch_train_history_agg)
        summary = "Train averages -> LR: {lr:.01e}, Loss: {loss:02.02f}, Acc: {accuracy:01.02}, TP {TP:02.0f}, FP {FP:02.0f}, TN {TN:02.0f}, FN {FN:02.0f}"
        summary = summary.format(**epoch_train_history_agg.to_dict())
        logger.info(summary)
        print(summary)
            
        # Validation loop
        model.eval()
        dataset.train = False # Don't augment validation/test data
        val_loop = tqdm(val_loader, leave=False, desc="Validation")
        epoch_val_history = []
        for batch_index, batch in enumerate(val_loop):
            image = batch['image'].float().to(DEVICE)
            label = batch['pain'].to(DEVICE)
            pred = model(image)
            epoch_val_history.append(get_stats(pred, label))
        
        epoch_val_history = pd.DataFrame(epoch_val_history)
        epoch_val_history_agg = epoch_val_history.agg({'accuracy': 'mean', 'TP': 'sum', 'FP': 'sum', 'TN': 'sum', 'FN': 'sum'})
        epoch_val_history_agg['epoch'] = epoch
        epoch_val_history_agg['shuffle'] = shuffle
        val_history.append(epoch_val_history_agg)
        summary = "Validation averages -> Acc: {accuracy:01.02}, TP {TP:02.0f}, FP {FP:02.0f}, TN {TN:02.0f}, FN {FN:02.0f}"
        summary = summary.format(**epoch_val_history_agg.to_dict())
        logger.info(summary)
        print(summary)
        
    # Create plots / save model
    train_history = pd.DataFrame(train_history)
    val_history = pd.DataFrame(val_history)
    
    train_history.to_csv(shuffle_dir / "train.csv", index=False)
    val_history.to_csv(shuffle_dir / "val.csv", index=False)
    
    if args.save_model: 
        torch.save(model.state_dict(), shuffle_dir / 'final_weights.pt')