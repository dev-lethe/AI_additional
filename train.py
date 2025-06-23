import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import MNIST_NN

### hyper parameters
bs = 32
lr = 0.002
epochs = 10
####

### dataset
train_data = torchvision.datasets.MNIST(train=True)
test_data = torchvision.datasets.MNIST(train=False)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bs)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs)
####

### model
model = MNIST_NN()
####

def training(
        model=model,
        dataloader=train_dataloader
        lr=lr,
        epochs=epochs,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
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