from pathlib import Path
from skimage.transform import resize
from skimage.io import imread
import torch
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomAffine
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm

class BaselineAugmentation:
    def __init__(self, degrees=(0, 30), translate=(0, 0.2), scale=(0.9, 1.0)):
        self.transform = Compose([
            RandomAffine(degrees=degrees, translate=translate, scale=scale),
            RandomHorizontalFlip()
        ])
        
    def __call__(self, x):
        return self.transform(x)

class BMv1(Dataset):
    """PyTorch Dataset Wrapper for the Black Mouse Dataset [1]
    
    References
    ----------
    [1] Hohlbaum, K., Andresen, N., Wöllhaf, M., Lewejohann, L., Hellwich, O., 
    Thöne-Reineke, C., & Belik, V. (2019). Black Mice Dataset v1.
    """
    def __init__(self, path, load_into_memory=False, training_transform=None):
        super().__init__()
        self.path = Path(path)
        
        # Used to decide whether to augment images or not
        self.train = True 
        self.training_transform = training_transform
        
        # Get images
        images = sorted(self.path.glob('images_224x224/*.jpg'))
        self.images = [p.absolute() for p in images]
        
        self.load_into_memory = load_into_memory
        if self.load_into_memory:
            with mp.Pool(mp.cpu_count()) as pool:
                self.images = list(tqdm(pool.imap(imread, self.images), total=len(images)))
            
        # Get labels
        self.labels = pd.read_csv(self.path / 'labels.csv')
        
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        # Get the image
        image = self.images[idx]
        if not self.load_into_memory:
            image = imread(self.images[idx])
            
        # Apply training transform 
        if self.train and self.training_transform:
            image = torch.from_numpy(image)
            image = self.training_transform(image)
            
        image = image.transpose(2, 0, 1).astype(np.float32)
        labels = self.labels.iloc[idx].to_dict()
        return {
            'image': image,
            **labels
        }
        
    