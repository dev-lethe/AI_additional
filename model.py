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
        x = input.reshape([784, 1])
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        pred = nn.Softmax(x)
        return pred