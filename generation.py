import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from model import MNIST_NN
from loss import tv_loss, binarization_loss


def generation(
    target=6,
    lr=0.001,
    epochs=5000,
    conf=False
):
    label = torch.tensor([target])
    img_tensor = torch.rand(1, 784, requires_grad=True)

    optimizer = torch.optim.Adam([img_tensor], lr=lr)
    criterion = nn.CrossEntropyLoss()

    model = MNIST_NN()
    ckpt = torch.load("/home/lethe/AI/data/model.pt")
    model.load_state_dict(ckpt["model"])
    model.eval()

    bar = tqdm(range(epochs))
    for i in bar:
        img_tensor.data.clamp_(0.0, 1.0)
        optimizer.zero_grad()
        output = model(img_tensor)
        loss = criterion(output, label)
        tv = tv_loss(img_tensor, weight=5e-3)
        bi = binarization_loss(img_tensor, weight=1e-3)
        bar.set_postfix(loss=f"{loss.item():.4f}", tv_loss=f"{tv.item():.4f}", bi_loss=f"{bi.item():.4f}")
        loss = 1e-1*loss + tv + bi
        loss.backward()
        optimizer.step()

    if conf:
        pred = model(img_tensor)
        _, lbl = torch.max(pred, 1)
        print(f"predict label: {lbl}")
        

    img = img_tensor.detach().view(28, 28)
    SAVE_PATH = f"./{target}.png"
    torchvision.utils.save_image(img.unsqueeze(0), SAVE_PATH)

    print(f"saved image >> {SAVE_PATH}")

if __name__ == "__main__":
    generation(conf=True)
