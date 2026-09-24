import torch
from torch.utils.data import DataLoader, RandomSampler
from brane_generator.dataset import BraneDataset
from torchvision.transforms import v2
import torchvision.utils as vis_utils
import matplotlib.pyplot as plt
from brane_generator.generator import Generator, Discriminator
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from scipy import linalg
import argparse
IS_RANGPUR = False
if not IS_RANGPUR:
    plt.switch_backend("module://kitcat")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ap = argparse.ArgumentParser()
ap.add_argument("chartoutput", type=str, default="charts/GAN.png")

image_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    ])

BATCH_SIZE = 48
NOISE_DIM = 128
OUTPUT_SIDE = 256
OUTPUT_CHANNELS = 1
LEARNING_RATE = 1e-3
VIS_BATCH = 9
VIS_ROWS = int(np.sqrt(VIS_BATCH))

# GAN metrics implementations adapted from https://medium.com/@heyamit10/pytorch-implementation-of-common-gan-metrics-86f993f6e737
def calc_frechet_inception(real_logits, fake_logits):
    r_mu = real_logits.mean(axis=0)
    r_sigma = torch.cov(real_logits, correction=0)
    f_mu = fake_logits.mean(axis=0)
    f_sigma = torch.cov(fake_logits, correction=0)
    covmean = torch.sqrt(r_sigma @ f_sigma)
    diff = r_mu - f_mu
    return diff.dot(diff) + torch.trace(f_sigma + r_sigma - 2* covmean)

def calc_kernel_inception(real_logits, fake_logits):
    gamma = 1.0/real_logits.shape[1]
    coef = 1
    degree = 3
    kernel_rr = (gamma * real_logits @ real_logits.T + coef) ** degree
    kernel_ff = (gamma * fake_logits @ fake_logits.T + coef) ** degree
    kernel_rf = (gamma * real_logits @ fake_logits.T + coef) ** degree
    return kernel_rr.mean() + kernel_ff.mean() - 2*kernel_rf.mean()

NUM_EPOCHS = 100

IS_REAL = 1.0
IS_FAKE = 0.0

train = BraneDataset("keras_png_slices_data", transform=image_transform)
trainloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
sampler = RandomSampler(train, num_samples=BATCH_SIZE)
randomsampler = DataLoader(train, sampler=sampler, batch_size=BATCH_SIZE, num_workers=8)
if __name__ == "__main__":
    args = ap.parse_args()
    generator = Generator(NOISE_DIM, OUTPUT_SIDE, OUTPUT_CHANNELS).to(device)
    discriminator = Discriminator(OUTPUT_SIDE, OUTPUT_CHANNELS).to(device)
    loss = nn.BCEWithLogitsLoss()
    optim_gen = optim.AdamW(generator.parameters(), lr=LEARNING_RATE*5)
    optim_dis = optim.AdamW(discriminator.parameters(), lr=LEARNING_RATE)
    gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_gen, T_max=NUM_EPOCHS, eta_min=1e-6)
    dis_scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_dis, T_max=NUM_EPOCHS, eta_min=1e-7)

    print(f"NUM GEN PARAMS: {sum(p.numel() for p in generator.parameters())}\nNUM DIS PARAMS: {sum(p.numel() for p in discriminator.parameters())}")

    num_steps = 0
    real_losses = []
    fake_losses = []
    gen_losses = []
    d_xs = []
    d_gx1s = []
    d_gx2s = []
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
            noise = torch.randn(images.shape[0], NOISE_DIM, device=device)
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
        dis_scheduler.step()
        gen_scheduler.step()
        with torch.no_grad():
            generator.eval()
            discriminator.eval()
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
            fake_image = generator(noise)
            real_image = next(iter(randomsampler)).to(device)
            if not IS_RANGPUR:
                fig, ax = plt.subplots(1, 2)
                fig.tight_layout()
                fig.set_dpi(100)
                fig.set_size_inches(16, 10)
                ax[0].imshow(
                    np.transpose(vis_utils.make_grid(fake_image[:VIS_BATCH], padding=2, normalize=True, nrow=VIS_ROWS).cpu(),
                    (1,2,0))
                )
                ax[0].set_xticks([])
                ax[0].set_yticks([])
                ax[1].imshow(
                    np.transpose(vis_utils.make_grid(real_image[:VIS_BATCH], padding=2, normalize=True, nrow=VIS_ROWS).cpu(),
                    (1,2,0))
                )
                ax[1].set_xticks([])
                ax[1].set_yticks([])
                plt.show()
            fake_logits = discriminator(fake_image)
            real_logits = discriminator(real_image)
            fid = calc_frechet_inception(real_logits, fake_logits)
            rid = calc_kernel_inception(real_logits, fake_logits)
            print(f"""EPOCH {e+1}/{NUM_EPOCHS}
Discriminator Loss:
Real Loss: {real_loss_sum/batches_done:.5f}
Fake Loss: {fake_loss_sum/batches_done:.5f}
Generator Loss: {gen_loss_sum/batches_done:.5f}
Avg discriminator prediction on real images: {D_x_sum/batches_done:.4f}
Avg discriminator prediction on fake images: {D_gx_1_sum/batches_done:.4f} / {D_gx_2_sum/batches_done:.4f}
Frechet Inception Distance: {fid:.5f}
Kernel Inception Distance: {rid:.5f}""")
            generator.train()
            discriminator.train()

        real_losses.append(real_loss_sum/batches_done)
        fake_losses.append(fake_loss_sum/batches_done)
        gen_losses.append(gen_loss_sum/batches_done)
        d_xs.append(D_x_sum/batches_done)
        d_gx1s.append(D_gx_1_sum/batches_done)
        d_gx2s.append(D_gx_2_sum/batches_done)

    fig, ax = plt.subplots(1, 2)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
    ax[0].plot(real_losses, label="Discriminator Real Loss")
    ax[0].plot(fake_losses, label="Discriminator Fake Loss")
    ax[0].plot(gen_losses, label="Generator Loss")
    ax[0].legend()
    ax[0].set_ylabel("Cross Entropy Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_title("Loss")

    ax[1].plot(d_xs, label="D(x)")
    ax[1].plot(d_gx1s, label="D(g(x)) (1)")
    ax[1].plot(d_gx2s, label="D(g(x)) (2)")
    ax[1].legend()
    ax[1].set_ylabel("Ratio of Correct Predictions")
    ax[1].set_xlabel("Epochs")
    ax[1].set_title("Ratio of Correct Predictions")
    fig.savefig(args.chartoutput) 
    torch.save(generator.state_dict(), "generator.ckpt")
