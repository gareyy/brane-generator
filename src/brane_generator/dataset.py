import os
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import decode_image
from enum import Enum

TEST_SLICES = "keras_png_slices_test"
TRAIN_SLICES = "keras_png_slices_train"
VALID_SLICES = "keras_png_slices_validate"

class dataslice(Enum):
    CTEST = "test"
    CTRAIN = "train"
    CVALID = "valid"

class BraneDataset(Dataset):
    def __init__(self, images_dir, ds: dataslice, transform=None) -> None:
        assert ds in [dataslice.CTEST, dataslice.CTRAIN, dataslice.CVALID], "Invalid data type!"
        self.root_dir = images_dir
        match ds:
            case dataslice.CTEST:
                self.images_dir = os.path.join(self.root_dir, TEST_SLICES)
            case dataslice.CTRAIN:
                self.images_dir = os.path.join(self.root_dir, TRAIN_SLICES)
            case _:
                self.images_dir = os.path.join(self.root_dir, VALID_SLICES)
        self.transform = transform
        self.imagelist = os.listdir(self.images_dir)
        self.dataslice = ds

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.imagelist[index])
        image = decode_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image
