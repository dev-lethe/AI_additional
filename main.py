import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os

from model import MNIST_NN, MNIST_CNN
from train import training, evaluation
from generation import generation


# hyper parameters
bs = 64
lr = 0.002
epochs = 7
target = 3

layer_dim = 1024

ch = 32
dim = 512

CONV = True
LOAD = False

if CONV:
    model = MNIST_CNN(channel=ch, dim=dim)
    SAVENAME = "cnn_model.pt"
    print("use cnn model")
else:
    model = MNIST_NN(layer_dim=layer_dim)
    SAVENAME = "nn_model.pt"
    print("use nn model")

if LOAD:
    path = os.path.join("/home/lethe/AI/data/", SAVENAME)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    print(f"loaded model << {path}")
else:
    train_data = torchvision.datasets.MNIST(root="/home/lethe/AI/data/train", train=True, transform=transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bs)

    training(
        model=model,
        dataloader=train_dataloader,
        lr=lr,
        epochs=epochs,
        save=True,
        SAVENAME=SAVENAME
    )

    test_data = torchvision.datasets.MNIST(root="/home/lethe/AI/data/test", train=False, transform=transforms.ToTensor())
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs)

    acc = evaluation(
        model=model,
        dataloader=test_dataloader,
    )

print(f"target: {target}")
generation(
    model,
    target=target,
    lr=0.001,
    lw=1e5,
    tvw=5e-4,
    biw=5e-4,
    epochs=30000,
    conf=True,
    CONV=CONV
)