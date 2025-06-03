from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        with gzip.open(image_filename, 'rb') as f:
            _, count, row, col = struct.unpack('>IIII', f.read(16))
            image_data = f.read(count * row * col)
        images = np.frombuffer(image_data, np.uint8)
        # Reshape to (count, row, col, 1) for H x W x C format
        self.images = np.reshape(images, (count, row, col, 1)).astype(np.float32) / 255.0
        
        with gzip.open(label_filename, 'rb') as f:
            _, count = struct.unpack('>II', f.read(8))
            label_data = f.read(count)
        self.labels = np.frombuffer(label_data,np.uint8)
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        img = self.images[index]
        label = self.labels[index]
        
        if self.transforms:
            for transform in self.transforms:
                img = transform(img)
        
        return img, label

    def __len__(self) -> int:
        return len(self.images)