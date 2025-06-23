import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model import MNIST_NN
from train import training, evaluation

if __name__ == "__main__":
    # hyper parameters
    bs = 64
    lr = 0.002
    epochs = 5

    # dataset
    train_data = torchvision.datasets.MNIST(root="/home/lethe/AI/data/train", train=True, transform=transforms.ToTensor())
    test_data = torchvision.datasets.MNIST(root="/home/lethe/AI/data/test", train=False, transform=transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bs)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs)

    # model
    model = MNIST_NN()
    
    # train
    """
    training(
        model=model,
        dataloader=train_dataloader,
        lr=lr,
        epochs=epochs,
        save=True
    )
    """
    acc = evaluation(
        model=model,
        dataloader=test_dataloader,
        load=True
    )
    print(f"accuracy: {acc}")