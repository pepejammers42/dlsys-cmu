import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        imgs = []
        labs = []
        for f in os.listdir(base_folder):
            if f.startswith('data' if train else 'test'):
                path = os.path.join(base_folder, f)
                with open(path, 'rb') as fo:
                    dict = pickle.load(fo, encoding='bytes')
                    imgs.append(dict[b'data'] / 255)
                    labs.append(dict[b'labels'])
        self.X = np.concatenate(imgs)
        self.y = np.concatenate(labs)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        return self.X[index].reshape(3,32,32), self.y[index]

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.y.shape[0]
