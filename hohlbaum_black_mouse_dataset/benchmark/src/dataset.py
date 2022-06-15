from pathlib import Path
from skimage.transform import resize
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

class BMv1(Dataset):
    """PyTorch Dataset Wrapper for the Black Mouse Dataset [1]
    
    References
    ----------
    [1] Hohlbaum, K., Andresen, N., Wöllhaf, M., Lewejohann, L., Hellwich, O., 
    Thöne-Reineke, C., & Belik, V. (2019). Black Mice Dataset v1.
    """
    def __init__(self, path, load_into_memory=False):
        super().__init__()
        self.path = Path(path)
        images = sorted(self.path.glob('images_224x224/*.jpg'))
        self.images = [p.absolute() for p in images]
        
        self.load_into_memory = load_into_memory
        if self.load_into_memory:
            with mp.Pool(mp.cpu_count()) as pool:
                self.images = list(tqdm(pool.imap(imread, self.images), total=len(images)))
            
        self.labels = pd.read_csv(self.path / 'labels.csv')
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if not self.load_into_memory:
            image = imread(self.images[idx])
            
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
        return train_sampler, val_sampler
    