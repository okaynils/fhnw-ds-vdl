from torch.utils.data import Subset
import numpy as np

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

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Total size of the dataset
    total_size = len(dataset)
    indices = np.arange(total_size)
    
    # Shuffle the indices
    np.random.shuffle(indices)
    
    # Calculate split points
    train_split = int(train_ratio * total_size)
    val_split = train_split + int(val_ratio * total_size)
    
    # Split the indices
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    # Create dataset subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset