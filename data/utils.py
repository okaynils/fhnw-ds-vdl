import torch
from torch.utils.data import Subset
import numpy as np
from PIL import ImageOps, Image

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_seed=42):
    """
    Splits the dataset into train, validation, and test partitions.

    :param dataset: The NYUDepthV2 dataset object.
    :param train_ratio: Proportion of the dataset to allocate to the training set.
    :param val_ratio: Proportion of the dataset to allocate to the validation set.
    :param test_ratio: Proportion of the dataset to allocate to the test set.
    :param random_seed: Seed for reproducibility of the split.
    :return: A tuple of (train_dataset, val_dataset, test_dataset).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    np.random.seed(random_seed)
    
    total_size = len(dataset)
    indices = np.arange(total_size)
    
    np.random.shuffle(indices)
    
    train_split = int(train_ratio * total_size)
    val_split = train_split + int(val_ratio * total_size)
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset

def unnormalize(img, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)  # Reshape to [C, 1, 1]
    std = torch.tensor(std).view(-1, 1, 1)    # Reshape to [C, 1, 1]
    img = std * img + mean
    img = torch.clip(img, 0, 1)
    return img

class HistogramEqualization:
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise ValueError("Input image must be a PIL image")
        return ImageOps.equalize(img)
