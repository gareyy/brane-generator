import torch.nn as nn

# inspired by https://www.geeksforgeeks.org/deep-learning/generative-adversarial-networks-gans-in-pytorch/ and https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html, and https://apxml.com/courses/cnns-for-computer-vision/chapter-7-gans-image-synthesis/implementing-dcgan-practice
KERNEL_SIZE = 4
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim, out_channels) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.out_channels = out_channels
        self.start_size = output_dim // 4 # we are using 3 upsample steps
        self.main = nn.Sequential(
            nn.Linear(self.noise_dim, self.start_size * self.start_size * 256),
            nn.Unflatten(1, (256, self.start_size, self.start_size)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            # upsample to start_size/2
            nn.ConvTranspose2d(256, 128, kernel_size=KERNEL_SIZE, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            # upsample to start_size
            nn.ConvTranspose2d(128, 64, kernel_size=KERNEL_SIZE, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            # output image with desired channels
            nn.ConvTranspose2d(64, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),

            nn.Sigmoid(), # image outputs are between 0 and 1
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, image_size, in_channels) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.main = nn.Sequential(
            # downsample to start_size/2
            nn.Conv2d(self.in_channels, 128, kernel_size=KERNEL_SIZE, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.2),
            # downsample to start_size/4
            nn.Conv2d(128, 64, kernel_size=KERNEL_SIZE, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.2),
            # downsample to start_size/8
            nn.Conv2d(64, 32, kernel_size=KERNEL_SIZE, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            nn.Linear(32 * self.image_size//8 * self.image_size // 8, 1, bias=False),
        )
    def forward(self, x):
        return self.main(x)
