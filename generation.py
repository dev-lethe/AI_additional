import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model import MNIST_NN


def generation(
    target=0,
    lr=0.001,
    epochs=10000
):
    label = torch.tensor([target])
    img_tensor = torch.rand(1, 784, requires_grad=True)

    optimizer = torch.optim.Adam([img_tensor], lr=lr)
    criterion = nn.CrossEntropyLoss()

    model = MNIST_NN()
    ckpt = torch.load("/home/lethe/AI/data/model.pt")
    model.load_state_dict(ckpt["model"])
    model.eval()

    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        output = model(img_tensor)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    img = img_tensor.detach().view(28, 28)
    SAVE_PATH = f"./{target}.png"
    torchvision.utils.save_image(img.unsqueeze(0), SAVE_PATH)

    print(f"saved image >> {SAVE_PATH}")

if __name__ == "__main__":
    generation()
