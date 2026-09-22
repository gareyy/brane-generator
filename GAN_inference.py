from brane_generator.generator import Generator
import GAN
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vis_utils
import numpy as np
plt.switch_backend("module://kitcat")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_IMAGES = 36
NUM_ROWS = int(np.sqrt(NUM_IMAGES))
if __name__ == "__main__":
    model = Generator(GAN.NOISE_DIM, GAN.OUTPUT_SIDE, GAN.OUTPUT_CHANNELS).to(device)
    model.load_state_dict(torch.load("generator.ckpt"))
    model.eval()
    with torch.no_grad():
        noise = torch.randn(NUM_IMAGES, GAN.NOISE_DIM, device=device)
        fake_image = model(noise)
        fig, ax = plt.subplots(1)
        fig.tight_layout()
        fig.set_dpi(200)
        ax.imshow(
            np.transpose(vis_utils.make_grid(fake_image[:NUM_IMAGES], padding=2, normalize=True, nrow=NUM_ROWS).cpu(),
            (1,2,0))
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

