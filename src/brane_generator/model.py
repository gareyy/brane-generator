import torch.nn as nn
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                )
    def forward(self, x):
        out = self.layers(x)
        return self.relu(out + x)

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                    nn.BatchNorm2d(out_channels),
                )
        self.layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                )
    def forward(self, x):
        out = self.layers(x)
        return self.relu(out + self.shortcut(x))

class ResnetEighteen(nn.Module):
    def __init__(self, num_classes, in_channels) -> None:
        super().__init__()
        self.layers = nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=1), # conv1
                    nn.BatchNorm2d(64),
                    nn.MaxPool2d(kernel_size=3, stride=2), #conv2_1
                    IdentityBlock(64, 64), #conv2_2
                    IdentityBlock(64, 64), #conv2_3
                    ConvolutionBlock(64, 128), #conv3_1
                    IdentityBlock(128, 128), #conv3_2
                    ConvolutionBlock(128, 256), #conv4_1
                    IdentityBlock(256, 256), #conv4_2
                    ConvolutionBlock(256, 512), #conv5_1
                    IdentityBlock(512, 512), #conv5_2
                    nn.AvgPool2d(kernel_size=1),
                    nn.Flatten(),
                    nn.Linear(512, 1000),
                    nn.Linear(1000, num_classes),
                )
    def forward(self, x):
        return self.layers(x)
