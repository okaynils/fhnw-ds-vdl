from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import os
import torch
import h5py
from PIL import Image
import numpy as np


class NYUDepthV2(Dataset):
    """
    PyTorch Dataset for the NYU Depth V2 dataset.
    Contains RGB images, semantic segmentation masks, depth maps, and instance masks.

    Returns:
    - RGB image: 3-channel input image
    - Semantic Segmentation: 1 channel representing class labels
    - Depth Image: 1 channel representing depth in meters
    - Instance Masks: A tensor of binary masks, one per object instance
    """

    BASE_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

    def __init__(self, root, download=False, preload=False, image_transform=None, seg_transform=None, depth_transform=None, n_classes=894):
        """
        Initialize the dataset and optionally download the dataset if not found locally.

        :param root: path to the dataset directory
        :param download: whether to download the dataset if it's not found
        :param image_transform: optional transformations for the RGB images
        :param seg_transform: optional transformations for the segmentation masks
        :param depth_transform: optional transformations for the depth maps
        :param instance_transform: optional transformations for the instance masks
        """
        super().__init__()
        self.root = root
        self.image_transform = image_transform
        self.seg_transform = seg_transform
        self.depth_transform = depth_transform
        self.preload = preload

        self.mat_file = os.path.join(root, "nyu_depth_v2_labeled.mat")
        if download:
            self.download()

        if not os.path.exists(self.mat_file):
            raise RuntimeError(f"Dataset not found at {self.mat_file}. Use download=True to download it.")

        self.data = None
        self.images = None
        self.segments = None
        self.depths = None
        self.names = None
        
        self.n_classes = n_classes

        if preload:
            self._load_data()

    def _load_data(self):
        if self.data is None:
            self.data = h5py.File(self.mat_file, 'r')
            self.images = self.data['images']
            self.segments = self.data['labels']
            self.depths = self.data['depths']
            self.names = self.data['names']

            # Resolve names dataset into a list of strings
            self.resolved_names = unpack_names(self.data, self.names)

    def __len__(self):
        self._load_data()
        return self.images.shape[0]

    def __getitem__(self, index):
        self._load_data()
        
        # Handle slicing
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self[i] for i in indices]
        
        # Load raw data for a single index
        img = self.images[index].transpose()
        seg = self.segments[index].transpose()
        depth = self.depths[index].transpose()

        # Convert to numpy arrays
        img = np.array(img)
        seg = np.array(seg)
        depth = np.array(depth)
        
        # Remove singleton dimensions
        if seg.ndim == 3 and seg.shape[0] == 1:
            seg = seg.squeeze(0)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth.squeeze(0)

        # Convert raw arrays to PIL images
        img = Image.fromarray(img)
        seg = Image.fromarray(seg.astype(np.uint16))
        depth = Image.fromarray(depth)

        # Apply transformations
        if self.image_transform is not None:
            img = self.image_transform(img)
        if self.seg_transform is not None:
            seg = self.seg_transform(seg)
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        # Convert transformed segmentation and depth maps back to numpy arrays
        seg = np.array(seg)
        depth = np.array(depth)

        # Extract class presence vector
        unique_classes = np.unique(seg)
        
        class_vector = np.zeros(self.n_classes, dtype=np.float16)
        class_vector[unique_classes] = 1

        # Compute depth vector
        depth_vector = np.zeros(self.n_classes, dtype=np.float32)
        for cls in unique_classes:
            class_mask = (seg == cls)
            if class_mask.sum() > 0:
                depth_vector[cls] = depth[class_mask].mean()

        return img, seg, depth, class_vector, depth_vector

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickleable entries
        state['data'] = None
        state['images'] = None
        state['segments'] = None
        state['depths'] = None
        
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reopen the file in the new process
        self._load_data()

    def download(self):
        """
        Downloads the NYU Depth V2 dataset .mat file and places it at the top level of the root directory.
        """
        if os.path.exists(self.mat_file):
            print("Dataset already exists, skipping download.")
            return
        
        print(f"Downloading dataset from {self.BASE_URL}...")
        download_url(self.BASE_URL, self.root)

        if not os.path.exists(self.mat_file):
            raise RuntimeError(f"Failed to download dataset. File not found at {self.mat_file}.")

        print("Download completed and dataset is ready for use.")
        
def unpack_names(file, names_dataset):
    """
    Unpack an HDF5 dataset of object references into a list of strings.

    :param file: The parent HDF5 file (h5py.File object).
    :param names_dataset: The HDF5 dataset containing object references.
    :return: A list of resolved names as strings.
    """
    resolved_names = []

    # Access the first element, which is an array of object references
    object_references = names_dataset[0]  # names_dataset has shape (1, 894)

    for ref in object_references:
        # Dereference the HDF5 object
        obj = file[ref]

        # Decode the array of ASCII values into a string
        ascii_array = obj[()]  # Get the numpy array of ASCII values
        if isinstance(ascii_array, np.ndarray):
            # Convert ASCII array to a string
            resolved_names.append("".join(chr(c[0]) for c in ascii_array))
        else:
            # Handle other data types (unlikely)
            resolved_names.append(str(ascii_array))

    return np.array(resolved_names)