"""Resize the images in BMv1 file to 224x224 for quicker training

Usage
-----
python resize_images.py /path/to/BMv1/Dataset
"""
import sys
import multiprocessing as mp

from functools import partial
from tqdm import tqdm
from pathlib import Path
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_ubyte

def resize_image(imagepath, output_dir, output_shape=(224, 224)):
    """Helper function for loading and reshaping images."""
    image = imread(imagepath)
    image = resize(image, output_shape)
    image = img_as_ubyte(image)
    
    output_path = output_dir / imagepath.parts[-1]
    imsave(output_path, image)
    return output_path
    
if __name__ == '__main__':
    dataset_path = Path(sys.argv[1]).absolute()
    assert dataset_path.exists(), "Argument must be a path to an existing dataset" 
    
    images = list(dataset_path.glob("images/*.jpg"))
    output_dir = dataset_path / 'images_224x224'
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    foo = partial(resize_image, output_dir=output_dir)
    with mp.Pool(mp.cpu_count()) as pool:
        images = list(tqdm(pool.imap_unordered(foo, images), total=len(images)))
