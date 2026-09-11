import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm
import time
from copy import deepcopy
from brane_generator.model import ResnetEighteen
import matplotlib.pyplot as plt

torch.manual_seed(67)

device = "cuda" if torch.cuda.is_available() else "cpu"

# from https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#stats = ((0.5, 0.5, 0.5), (0.5, 0.5,0.5))
train_transform = v2.Compose([
    v2.RandomHorizontalFlip(0.5),
    v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
    v2.RandomCrop(32, padding=4),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(*stats), # standardise values to range [-1, 1]
    ])

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(*stats), # standardise values to range [-1, 1]
    ])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

BATCH_SIZE = 512
EPOCHS = 1000 # limit for a100, but we have a time limit
DISABLE_TQDM = False # change in rangpur
VALID_RATIO = 0.1

if __name__ == "__main__":
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainset, validset = torch.utils.data.random_split(dataset, [1-VALID_RATIO, VALID_RATIO])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False)
    validloader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)
    
    model = ResnetEighteen(len(classes),3).to(device)
    print(model)
    best = deepcopy(model)
    best_valid_ratio = -1.0

    cel = nn.CrossEntropyLoss()
    optimiser = optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4, lr=0.1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimiser, epochs=EPOCHS, steps_per_epoch=len(trainloader), max_lr=0.1)

    start = time.time()
    last_epoch = time.time()

    train_losses = []
    valid_losses = []
    test_losses = []

    train_ratios = []
    valid_ratios = []
    test_ratios = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        for i, (inputs, labels) in tqdm(enumerate(trainloader), disable=DISABLE_TQDM, total=len(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = cel(logits, labels)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimiser.step()
            optimiser.zero_grad()
            train_loss += loss.item()
            num_batches += 1
            _, predictions = logits.max(1)
            total_predictions += inputs.shape[0]
            correct_predictions += predictions.eq(labels).sum().item()
            if scheduler:
                scheduler.step()
        train_loss /= num_batches
        print(f"EPOCH {epoch+1}")
        if scheduler:
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.8f}")
        print(f"TRAIN LOSS: {train_loss:.4f} Correct Predictions During Training: {correct_predictions}/{total_predictions}, {correct_predictions*100/total_predictions:.2f}%")
        train_losses.append(train_loss)
        train_ratios.append(correct_predictions*100/total_predictions)
        with torch.no_grad():
            valid_loss = 0.0
            num_batches = 0
            correct_predictions = 0
            total_predictions = 0
            for i, (inputs, labels) in enumerate(validloader):
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = cel(logits, labels)
                valid_loss += loss.item()
                num_batches += 1
                _, predictions = logits.max(1)
                total_predictions += inputs.shape[0]
                correct_predictions += predictions.eq(labels).sum().item()
        valid_loss /= num_batches
        ratio = correct_predictions*100/total_predictions
        valid_losses.append(valid_loss)
        valid_ratios.append(ratio)
        isbest = False
        if ratio > best_valid_ratio:
            best_valid_ratio = ratio
            best = deepcopy(model)
            isbest = True
        print(f"VALID LOSS: {valid_loss:.4f} Correct Predictions During Validation: {correct_predictions}/{total_predictions}, {ratio:.2f}%{' - Best Model!' if isbest else ''}")
        with torch.no_grad():
            test_loss = 0.0
            num_batches = 0
            correct_predictions = 0
            total_predictions = 0
            for i, (inputs, labels) in enumerate(testloader):
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = cel(logits, labels)
                test_loss += loss.item()
                num_batches += 1
                _, predictions = logits.max(1)
                total_predictions += inputs.shape[0]
                correct_predictions += predictions.eq(labels).sum().item()
        test_loss /= num_batches
        ratio = correct_predictions*100/total_predictions
        test_losses.append(test_loss)
        test_ratios.append(ratio)
        print(f"TEST LOSS: {test_loss:.4f} Correct Predictions During Testing: {correct_predictions}/{total_predictions}, {ratio:.2f}%")
        now = time.time()
        minutes_elapsed = (now-start)/60
        seconds_in_epoch = now-last_epoch
        last_epoch = time.time()
        print(f"MINUTES ELAPSED: {minutes_elapsed:.4f}m, Secs per Epoch: {seconds_in_epoch:.4f}s")
        if ratio > 90.0:
            break
        if minutes_elapsed >= 30:
            break

    print("DOING TEST DATASET")
    with torch.no_grad():
        test_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = cel(logits, labels)
            test_loss += loss.item()
            num_batches += 1
            _, predictions = logits.max(1)
            total_predictions += inputs.shape[0]
            correct_predictions += predictions.eq(labels).sum().item()
    test_loss /= num_batches
    ratio = correct_predictions*100/total_predictions
    print(f"TEST LOSS: {test_loss:.4f} Correct Predictions During Testing: {correct_predictions}/{total_predictions}, {ratio:.2f}%")

    fig, ax = plt.subplots(1, 2)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
    ax[0].plot(train_losses, label="Train Loss")
    ax[0].plot(valid_losses, label="Valid Loss")
    ax[0].plot(test_losses, label="Test Loss")
    ax[0].legend()
    ax[0].set_ylabel("Cross Entropy Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_title("Loss")

    ax[1].plot(train_ratios, label="Train Ratio")
    ax[1].plot(valid_ratios, label="Valid Ratio")
    ax[1].plot(test_ratios, label="Test Ratio")
    ax[1].legend()
    ax[1].set_ylabel("Ratio of Correct Predictions")
    ax[1].set_xlabel("Epochs")
    ax[1].set_title("Ratio of Correct Predictions")
    fig.savefig("charts/resnet18-cifar-fp.png") 
    torch.save(model.state_dict(), "./resnet18-fp.ckpt")
