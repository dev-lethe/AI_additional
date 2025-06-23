import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model import MNIST_NN
from train import training, evaluation
from generation import generation


# hyper parameters
bs = 64
lr = 0.002
epochs = 5
target = 7
LOAD = False

# dataset
train_data = torchvision.datasets.MNIST(root="/home/lethe/AI/data/train", train=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root="/home/lethe/AI/data/test", train=False, transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bs)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs)

# model
model = MNIST_NN()

if LOAD:
    ckpt = torch.load("/home/lethe/AI/data/model.pt")
    model.load_state_dict(ckpt["model"])
else:
    training(
        model=model,
        dataloader=train_dataloader,
        lr=lr,
        epochs=epochs,
        save=True
    )

acc = evaluation(
    model=model,
    dataloader=test_dataloader,
)
print(f"accuracy: {acc}")

generation(target=target)