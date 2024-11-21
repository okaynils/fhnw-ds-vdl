# Cell 2: Updated NYUDepthV2 Dataset class
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
    Contains RGB images, semantic segmentation masks, and depth maps.

    Returns:
    - RGB image: 3-channel input image
    - Semantic Segmentation: 1 channel representing class labels
    - Depth Image: 1 channel representing depth in meters
    """

    BASE_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

    def __init__(self, root, download=False, image_transform=None, seg_transform=None, depth_transform=None):
        """
        Initialize the dataset and optionally download the dataset if not found locally.

        :param root: path to the dataset directory
        :param download: whether to download the dataset if it's not found
        :param image_transform: optional transformations for the RGB images
        :param seg_transform: optional transformations for the segmentation masks
        :param depth_transform: optional transformations for the depth maps
        """
        super().__init__()
        self.root = root
        self.image_transform = image_transform
        self.seg_transform = seg_transform
        self.depth_transform = depth_transform

        self.mat_file = os.path.join(root, "nyu_depth_v2_labeled.mat")

        if download:
            self.download()

        if not os.path.exists(self.mat_file):
            raise RuntimeError(f"Dataset not found at {self.mat_file}. Use download=True to download it.")

        # Defer loading of the .mat file until after the worker process is initialized
        self.data = None
        self.images = None
        self.segments = None
        self.depths = None

    def _load_data(self):
        if self.data is None:
            self.data = h5py.File(self.mat_file, 'r')
            self.images = self.data['images']
            self.segments = self.data['labels']
            self.depths = self.data['depths']

    def __len__(self):
        self._load_data()
        return self.images.shape[0]

    def __getitem__(self, index):
        self._load_data()
        img = self.images[index].transpose(1, 2, 0)
        seg = self.segments[index]
        depth = self.depths[index]

        # Convert to numpy arrays
        img = np.array(img)
        seg = np.array(seg)
        depth = np.array(depth)

        # Remove singleton dimension
        if seg.ndim == 3 and seg.shape[0] == 1:
            seg = seg.squeeze(0)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth.squeeze(0)

        # Convert image to PIL Image and apply transformations
        img = Image.fromarray(img)
        img = img.rotate(-90)
        if self.image_transform is not None:
            img = self.image_transform(img)

        # Apply transformations to segmentation and depth if needed
        if self.seg_transform is not None:
            seg = Image.fromarray(seg.astype(np.uint8))
            seg = seg.rotate(-90)
            seg = self.seg_transform(seg)

        if self.depth_transform is not None:
            depth = Image.fromarray(depth)
            depth = depth.rotate(-90)
            depth = self.depth_transform(depth)

        return img, seg, depth


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
