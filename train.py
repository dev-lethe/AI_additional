import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

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
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
####

def training(
        model=model,
        dataloader=train_dataloader
        lr=lr,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer
):
    model.train()
    optimizer.zero_grad()
    for batch, (img, lbl) in enumerate(dataloader):
        pred = model(img)
        loss = criterion(pred, lbl)

        loss.backward()
        optimizer.step()