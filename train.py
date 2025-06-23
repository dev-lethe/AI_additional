import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import MNIST_NN

### hyper parameters
bs = 32
lr = 0.002
epochs = 10
####

### dataset
train_data = torchvision.datasets.MNIST(root="/home/lethe/AI/data/train", train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root="/home/lethe/AI/data/test", train=False, transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bs)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs)
####

### model
model = MNIST_NN()
####

def training(
        model=model,
        dataloader=train_dataloader,
        lr=lr,
        epochs=epochs,
        save=False
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"epoch: {epoch+1} / {epochs}"
        )

        for batch, (img, lbl) in progress_bar:
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, lbl)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    if save:
        SAVE_PATH = "/home/lethe/AI/data/model.pt"
        torch.save({"model": model.state_dict()}, SAVE_PATH)


def evaluation(
        model=model,
        dataloader=test_dataloader
):
    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="evaluating"
        )
        nums = 0
        correct = 0
        for batch, (img, lbl) in progress_bar:
            pred = model(img)
            _, pred_labels = torch.max(pred.data, 1)
            nums += pred_labels.size(0)
            correct += (pred_labels == lbl).sum().item()
    
    acc = 100 * correct / nums
    return acc