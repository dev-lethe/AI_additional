import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model import MNIST_NN, MNIST_CNN
from loss import tv_loss, binarization_loss


def generation(
    model,
    target=6,
    lr=0.05,
    epochs=10000,
    lw=1e3,
    tvw=3e-5,
    biw=1e-4,
    conf=False,
    CONV=False
):
    label = torch.tensor([target])
    img_tensor = torch.rand(1, 784, requires_grad=True)
    if CONV:
        img_tensor = torch.rand(1, 1, 28, 28, requires_grad=True)

    optimizer = torch.optim.Adam([img_tensor], lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.eval()

    bar = tqdm(range(epochs), desc="generating")
    for i in bar:
        img_tensor.data.clamp_(0.0, 1.0)
        optimizer.zero_grad()
        output = model(img_tensor)
        loss = criterion(output, label)
        tv = tv_loss(img_tensor, weight=tvw)
        bi = binarization_loss(img_tensor, weight=biw)
        bar.set_postfix(loss=f"{loss.item():.4f}", tv_loss=f"{tv.item():.4f}", bi_loss=f"{bi.item():.4f}")
        loss = lw*loss + tv + bi
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            img = img_tensor.detach().view(28, 28)
            SAVE_PATH = f"cnn/{target}.png"
            torchvision.utils.save_image(img.unsqueeze(0), SAVE_PATH)

    if conf:
        pred = model(img_tensor)
        _, lbl = torch.max(pred, 1)
        print(f"predict label: {lbl}")
        

    img = img_tensor.detach().view(28, 28)
    SAVE_PATH = f"nn/{target}.png"
    torchvision.utils.save_image(img.unsqueeze(0), SAVE_PATH)
    print(f"saved image >> {SAVE_PATH}")

if __name__ == "__main__":
    CONV = False
    model = MNIST_NN(layer_dim=1024)
    ckpt = torch.load("/home/lethe/AI/data/nn_model.pt")
    if CONV:
        model = MNIST_CNN(channel=32, dim=512)
        ckpt = torch.load("/home/lethe/AI/data/cnn_model.pt")
    model.load_state_dict(ckpt["model"])

    for target in range(10):
        generation(
            model,
            target=target,
            lr=0.005,
            lw=1e3,
            tvw=3e-5,
            biw=1e-5,
            epochs=10000,
            conf=True,
            CONV=CONV
        )
