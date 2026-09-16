import torch
from torch.utils.data import DataLoader
from brane_generator.dataset import BraneDataset, dataslice
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from brane_generator.resnet import ResnetDiscriminator
from brane_generator.generator import Generator
plt.switch_backend("module://kitcat")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    ])
"""
for reference: how to plot a tensor image
plt.imshow(np.transpose(inputtensor, (1, 2, 0)), cmap="grey")
"""
BATCH_SIZE = 512
NOISE_DIM = 100
OUTPUT_SIDE = 256
OUTPUT_CHANNELS = 3

train = BraneDataset("keras_png_slices_data", dataslice.CTRAIN, transform=image_transform)
trainloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
if __name__ == "__main__":
    gen = Generator(NOISE_DIM, OUTPUT_SIDE, OUTPUT_CHANNELS).to(device)
    discrim = ResnetDiscriminator(2, OUTPUT_CHANNELS).to(device)
    one = train.__getitem__(0)
    one = one.unsqueeze(0)
    print(one.shape)

    noise = torch.randn(1, NOISE_DIM, device=device)
    print(noise.shape)
    fake = gen(noise)
    print(fake.shape)

    pred = discrim(fake)
    print(pred.shape)
    print(pred)
