import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

# from https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # standardise values to range [-1, 1]

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
        out += self.relu(x)
        return self.relu(out)

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
        out += self.shortcut(x)
        return self.relu(out)

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

BATCH_SIZE = 4
EPOCHS = 1
DISABLE_TQDM = False # change in rangpur

if __name__ == "__main__":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = ResnetEighteen(len(classes),3).to(device)
    print(model)

    cel = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    start = time.time()
    torch.autograd.set_detect_anomaly(True, check_nan=False)
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        num_batches = 0
        for i, (inputs, labels) in tqdm(enumerate(trainloader), disable=DISABLE_TQDM, total=len(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = cel(logits, labels)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
            num_batches += 1
        train_loss /= num_batches
        print(f"EPOCH {epoch}, TRAIN LOSS: {train_loss}")
        now = time.time()
        print(f"{(now-start)/60:.2f}m")
