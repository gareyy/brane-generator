from brane_generator.resnet import ResnetEighteen
import torch
import matplotlib.pyplot as plt
import matplotlib
import torchvision.utils as vis_utils
import torchvision.transforms.v2 as v2
import torchvision
import torch.nn as nn
import numpy as np
plt.switch_backend("module://kitcat")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_IMAGES = 36
NUM_ROWS = int(np.sqrt(NUM_IMAGES))
font = {
        'weight' : 'bold',
        'size'   : 5}

matplotlib.rc('font', **font)

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
test_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(*stats), # standardise values to range [-1, 1]
    ])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=NUM_IMAGES, shuffle=True, num_workers=8, pin_memory=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
if __name__ == "__main__":
    model = ResnetEighteen(len(classes), 3).to(device)
    model.load_state_dict(torch.load("90resnet.ckpt"))
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(testloader))
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
    logits = nn.functional.softmax(logits, dim=-1)
    confs, predictions = logits.max(1)
    cls_preds = [classes[i] for i in predictions]
    cls_trues = [classes[i] for i in labels]
    fig, axes = plt.subplots(NUM_ROWS, NUM_ROWS)
    fig.tight_layout()
    fig.set_dpi(200)
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.transpose(
            vis_utils.make_grid(images[i].cpu(), normalize=True),
            (1,2,0))
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{cls_preds[i]}/{cls_trues[i]}\n{confs[i]*100:.2f}%")
        ax.figure.set_size_inches(7, 7)
    plt.show()
    print(f"Accuracy: {100*predictions.eq(labels).sum().item()/NUM_IMAGES:.3f}%")

