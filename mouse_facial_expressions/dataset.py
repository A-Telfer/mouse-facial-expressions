from pathlib import Path
from skimage.transform import resize
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('mouse_facial_expressions')

class BMv1(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.images = list(self.path.glob('images/*.jpg'))
        self.labels = pd.read_csv(self.path / 'labels.csv')
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        image = imread(self.path / self.images[idx])
        image = resize(image, (256, 256))
        image = image.transpose(2, 0, 1).astype(np.float32)
        labels = self.labels.iloc[idx].to_dict()
        
        return {
            'image': image,
            **labels
        }
        
    def train_test_split(self, train_ratio=0.9):
        """Train test split based on id"""
        ids = self.labels.id.unique()
        train_ids = set(np.random.choice(ids, int(len(ids) * train_ratio), replace=False))
        train_indices = self.labels[self.labels.id.isin(train_ids)].index.tolist()
        train_sampler = SubsetRandomSampler(train_indices)

        val_ids = set(ids) - train_ids
        val_indices = self.labels[self.labels.id.isin(val_ids)].index.tolist()
        val_sampler = SubsetRandomSampler(val_indices)
        
        logger.info(f"Train Ratio: {train_ratio}")
        logger.info(f"Train Indices: {train_indices}")
        logger.info(f"Validation Indices: {val_indices}")
        return train_sampler, val_sampler