import os
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import decode_image
from enum import Enum

TEST_SLICES = "keras_png_slices_test"
TRAIN_SLICES = "keras_png_slices_train"
VALID_SLICES = "keras_png_slices_validate"

class BraneDataset(Dataset):
    def __init__(self, images_dir, transform=None) -> None:
        self.root_dir = images_dir
        self.transform = transform
        self.imagelist = []
        for dirpath, _, filenames in os.walk(os.path.join(self.root_dir, TEST_SLICES)):
            for f in filenames:
                self.imagelist.append(os.path.abspath(os.path.join(dirpath, f)))
        for dirpath, _, filenames in os.walk(os.path.join(self.root_dir, TRAIN_SLICES)):
            for f in filenames:
                self.imagelist.append(os.path.abspath(os.path.join(dirpath, f)))
        for dirpath, _, filenames in os.walk(os.path.join(self.root_dir, VALID_SLICES)):
            for f in filenames:
                self.imagelist.append(os.path.abspath(os.path.join(dirpath, f)))

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, index):
        img_path = self.imagelist[index]
        image = decode_image(img_path, mode='GRAY')
        if self.transform:
            image = self.transform(image)
        return image
