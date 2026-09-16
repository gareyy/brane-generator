import torch
from brane_generator.dataset import BraneDataset, dataslice

train = BraneDataset("keras_png_slices_data", dataslice.CTRAIN)
print(len(train))
torch.set_printoptions(profile="full")
print(train.__getitem__(0))
