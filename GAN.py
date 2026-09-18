import torch
from torch.utils.data import DataLoader, RandomSampler
from brane_generator.dataset import BraneDataset, dataslice
from torchvision.transforms import v2
import torchvision.utils as vis_utils
import matplotlib.pyplot as plt
from brane_generator.generator import Generator, Discriminator
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
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
BATCH_SIZE = 64
NOISE_DIM = 128
OUTPUT_SIDE = 256
OUTPUT_CHANNELS = 1
LEARNING_RATE = 1e-2
VIS_BATCH = 9
VIS_ROWS = int(np.sqrt(VIS_BATCH))

NUM_EPOCHS = 10

IS_REAL = 1.0
IS_FAKE = 0.0

train = BraneDataset("keras_png_slices_data", dataslice.CTRAIN, transform=image_transform)
trainloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
sampler = RandomSampler(train, num_samples=VIS_BATCH)
randomsampler = DataLoader(train, sampler=sampler, batch_size=VIS_BATCH, num_workers=8)
if __name__ == "__main__":
    generator = Generator(NOISE_DIM, OUTPUT_SIDE, OUTPUT_CHANNELS).to(device)
    discriminator = Discriminator(OUTPUT_SIDE, OUTPUT_CHANNELS).to(device)
    loss = nn.BCEWithLogitsLoss()
    optim_gen = optim.AdamW(generator.parameters(), lr=LEARNING_RATE)
    optim_dis = optim.AdamW(discriminator.parameters(), lr=LEARNING_RATE)

    num_steps = 0
    for e in range(NUM_EPOCHS):
        real_loss_sum = 0.0
        fake_loss_sum = 0.0
        gen_loss_sum = 0.0
        D_x_sum = 0.0
        D_gx_1_sum = 0.0
        D_gx_2_sum = 0.0
        batches_done = 0
        for i, images in tqdm(enumerate(trainloader), total=len(trainloader), disable=False):
            # updating Discriminator
            # real batch train
            discriminator.zero_grad()
            real_images = images.to(device)
            labels = torch.full((images.shape[0], 1,), IS_REAL, dtype=torch.float32, device=device)
            outputs = discriminator(real_images)
            D_x = nn.functional.sigmoid(outputs).mean().item()
            real_loss = loss(outputs, labels)
            real_loss.backward()

            # fake batch train
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
            fake_images = generator(noise)
            labels.fill_(IS_FAKE)
            outputs = discriminator(fake_images.detach())
            D_gx_1 = nn.functional.sigmoid(outputs).mean().item()
            fake_loss = loss(outputs, labels)
            fake_loss.backward()
            optim_dis.step()

            # updating Generator
            generator.zero_grad()
            labels.fill_(IS_REAL)
            outputs = discriminator(fake_images)
            D_gx_2 = nn.functional.sigmoid(outputs).mean().item()
            gen_loss = loss(outputs, labels)
            gen_loss.backward()
            optim_gen.step()

            real_loss_sum += real_loss.detach().item()
            fake_loss_sum += fake_loss.detach().item()
            gen_loss_sum += gen_loss.detach().item()
            D_x_sum += D_x
            D_gx_1_sum += D_gx_1
            D_gx_2_sum += D_gx_2
            batches_done += 1
            #print(f"{real_loss.detach().item()}, {fake_loss.detach().item()}, {gen_loss.detach().item()}, {D_x}, {D_gx_1}, {D_gx_2}")
        print(f"""EPOCH {e+1}/{NUM_EPOCHS}
Discriminator Loss:
Real Loss: {real_loss_sum/batches_done:.5f}
Fake Loss: {fake_loss_sum/batches_done:.5f}
Generator Loss: {gen_loss_sum/batches_done:.5f}
Avg discriminator prediction on real images: {D_x_sum/batches_done:.4f}
Avg discriminator prediction on fake images: {D_gx_1_sum/batches_done:.4f} / {D_gx_2_sum/batches_done:.4f}""")
        with torch.no_grad():
            generator.eval()
            noise = torch.randn(VIS_BATCH, NOISE_DIM, device=device)
            fake_image = generator(noise)
            fig, ax = plt.subplots(1, 2)
            fig.tight_layout()
            ax[0].imshow(np.transpose(vis_utils.make_grid(fake_image[:VIS_BATCH], padding=2, normalize=True, nrow=VIS_ROWS).cpu(),(1,2,0)))
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            real_image = next(iter(randomsampler))
            ax[1].imshow(np.transpose(vis_utils.make_grid(real_image[:VIS_BATCH], padding=2, normalize=True, nrow=VIS_ROWS).cpu(),(1,2,0)))
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            plt.show()
        generator.train()
