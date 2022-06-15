from pathlib import Path
from skimage.transform import resize
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import logging
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from tqdm import tqdm
logger = logging.getLogger('mouse_facial_expressions')

def load_image(imagepath, output_shape=(224, 224)):
    """Helper function for loading and reshaping images."""
    return resize(imread(imagepath), output_shape)
        
class BMv1(Dataset):
    """PyTorch Dataset Wrapper for the Black Mouse Dataset [1]
    
    References
    ----------
    [1] Hohlbaum, K., Andresen, N., Wöllhaf, M., Lewejohann, L., Hellwich, O., 
    Thöne-Reineke, C., & Belik, V. (2019). Black Mice Dataset v1.
    """
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        images = sorted(self.path.glob('images/*.jpg'))
        images = [p.absolute() for p in images]
        with mp.Pool(mp.cpu_count()) as pool:
            self.images = list(tqdm(pool.imap(load_image, images), total=len(images)))
            
        self.labels = pd.read_csv(self.path / 'labels.csv')
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        image = self.images[idx].transpose(2, 0, 1).astype(np.float32)
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
        # logger.info(f"Train Indices: {train_indices}")
        # logger.info(f"Validation Indices: {val_indices}")
        return train_sampler, val_sampler
    
    def save(self, save_path='/datasets/BMv1.pk'):
        save_path = Path(save_path)
        if not save_path.exists():
            save_path.parent.mkdir(parents=True)
            
        with open(save_path, 'wb') as fp:
            pickle.dump(dataset, fp)
    
    @staticmethod
    def load(save_path='/datasets/BMv1.pk'):
        with open(save_path, 'rb') as fp:
            return pickle.load(fp)
        
        
# Load and pickle the dataset if this file is run directly
# Takes one argument, the filepath of the dataset
if __name__ == '__main__':
    import pickle
    import sys
    
    dataset_path = Path(sys.argv[1]).absolute()
    assert dataset_path.exists(), "Argument must be a path to an existing dataset" 
    dataset = BMv1(dataset_path)
    dataset.save()