import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm
import time
from copy import deepcopy
from brane_generator.model import ResnetEighteen

torch.manual_seed(67)

device = "cuda" if torch.cuda.is_available() else "cpu"

# from https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#stats = ((0.5, 0.5, 0.5), (0.5, 0.5,0.5))
# TODO: explain data augmentation
train_transform = v2.Compose([
    #v2.Pad(padding=4),
    #v2.RandomResizedCrop(size=(32, 32), antialias=True),
    v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(*stats)]) # standardise values to range [-1, 1]

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(*stats)]) # standardise values to range [-1, 1]

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#BATCH_SIZE = 8192
BATCH_SIZE = 1024
EPOCHS = 200
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
    optimiser = optim.AdamW(model.parameters(), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    start = time.time()
    torch.autograd.set_detect_anomaly(True, check_nan=False)
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
            #nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimiser.step()
            optimiser.zero_grad()
            train_loss += loss.item()
            num_batches += 1
            _, predictions = logits.max(1)
            total_predictions += inputs.shape[0]
            correct_predictions += predictions.eq(labels).sum().item()
        train_loss /= num_batches
        print(f"EPOCH {epoch}, TRAIN LOSS: {train_loss:.4f}")
        print(f"Correct Predictions During Training: {correct_predictions}/{total_predictions}, {correct_predictions*100/total_predictions:.2f}%")
        scheduler.step()
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
        isbest = False
        if ratio < best_valid_ratio:
            best_valid_ratio = ratio
            best = deepcopy(model)
            isbest = True
        print(f"VALID LOSS: {valid_loss:.4f} Correct Predictions During Validation: {correct_predictions}/{total_predictions}, {ratio:.2f}%{' - Best Model!' if isbest else ''}")
        now = time.time()
        minutes_elapsed = (now-start)/60
        print(f"{minutes_elapsed:.2f}m")
        # TESTING, REMOVE LATER
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
        # TESTING, REMOVE LATER
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
    torch.save(model.state_dict(), "./resnet18-fp.ckpt")
