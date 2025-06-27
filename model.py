import torch
import torch.nn as nn

class MNIST_NN(nn.Module):
    def __init__(
            self,
            layer_dim=1024,
    ):
        super().__init__()
        self.layer1 = nn.Linear(784, layer_dim)
        self.layer2 = nn.Linear(layer_dim, layer_dim)
        self.layer3 = nn.Linear(layer_dim, 10)
        self.act = nn.ReLU()

    def forward(self, input):
        x = input.view(input.size(0), -1) 
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        return x
    
class MNIST_CNN(nn.Module):
    def __init__(
            self,
            channel=64,
            dim=128
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channel, kernel_size=[3,3], padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=[3,3], padding=1)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=[3,3], padding=1)
        self.linear1 = nn.Linear(channel*7*7, dim)
        self.linear2 = nn.Linear(dim, 10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.act(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.act(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x