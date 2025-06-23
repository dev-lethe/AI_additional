import torch

def tv_loss(img, weight=1e-4):
    img = img.view(28, 28)
    h = torch.pow(img[:, :-1] - img[:, 1:], 2).sum()
    v = torch.pow(img[:-1, :] - img[1:, :], 2).sum()
    tv = weight * (h + v)
    return tv

def binarization_loss(img, weight=1e-4):
    img = img.view(28, 28)
    bi = weight * torch.sum(img * (1 - img))
    return bi