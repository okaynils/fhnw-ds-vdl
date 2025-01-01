import torch

def unnormalize(img, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    img = std * img + mean
    img = torch.clip(img, 0, 1)
    return img