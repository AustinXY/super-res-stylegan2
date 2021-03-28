from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils
import pickle
import random
import torch
import numpy as np

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            value = txn.get(key)

        im_pair = pickle.loads(value)
        buffer = im_pair.get_image_buffer()
        img = Image.open(buffer)
        mask = im_pair.mask

        img = self.transform(img)
        mask = transforms.ToTensor()(np.array(mask))

        if torch.rand(1) > 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)

        return img, mask
