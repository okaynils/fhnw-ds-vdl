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

    def __init__(self, root, download=False, image_transform=None, seg_transform=None, depth_transform=None, instance_transform=None):
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
        self.instance_transform = instance_transform

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
        self.instances = None

    def _load_data(self):
        if self.data is None:
            self.data = h5py.File(self.mat_file, 'r')
            self.images = self.data['images']
            self.segments = self.data['labels']
            self.depths = self.data['depths']
            # self.instances = self.data['instances']

    def __len__(self):
        self._load_data()
        return self.images.shape[0]

    def __getitem__(self, index):
        self._load_data()
        img = self.images[index].transpose()
        seg = self.segments[index].transpose()
        depth = self.depths[index].transpose()
        #instances = self.instances[index].transpose()

        # Convert to numpy arrays
        img = np.array(img)
        seg = np.array(seg)
        depth = np.array(depth)
        #instances = np.array(instances)

        # Remove singleton dimension
        if seg.ndim == 3 and seg.shape[0] == 1:
            seg = seg.squeeze(0)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth.squeeze(0)

        # Convert image to PIL Image and apply transformations
        img = Image.fromarray(img)
        if self.image_transform is not None:
            img = self.image_transform(img)

        # Apply transformations to segmentation and depth if needed
        if self.seg_transform is not None:
            seg = Image.fromarray(seg.astype(np.uint8))
            seg = self.seg_transform(seg)

        if self.depth_transform is not None:
            depth = Image.fromarray(depth)
            depth = self.depth_transform(depth)
            
        # instance_masks = get_instance_masks(instances)
        # instance_masks = [torch.tensor(mask) for mask in instance_masks]  # Convert to PyTorch tensors

        return img, seg, depth


    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickleable entries
        state['data'] = None
        state['images'] = None
        state['segments'] = None
        state['depths'] = None
        #state['instances'] = None
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

def get_instance_masks(instances):
    """
    Mimics the behavior of MATLAB's get_instance_masks.m to generate binary masks.

    :param instances: HxWxN or HxW numpy array of instance maps from the dataset.
    :return: List of binary masks, one for each object instance.
    """
    if instances.ndim == 2:
        # Convert HxW to HxWx1 for consistent processing
        instances = np.expand_dims(instances, axis=-1)

    H, W, N = instances.shape
    instance_masks = []

    # Iterate over the third dimension (N slices)
    for i in range(N):
        instance_slice = instances[:, :, i]
        unique_ids = np.unique(instance_slice)

        # Skip if the slice has no valid IDs
        if len(unique_ids) == 1 and unique_ids[0] == 0:
            continue

        for obj_id in unique_ids:
            if obj_id == 0:  # Ignore background
                continue

            # Create binary mask for this object ID
            binary_mask = (instance_slice == obj_id).astype(np.uint8)
            instance_masks.append(binary_mask)

    return instance_masks
