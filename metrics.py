import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

def get_features(images, inception_model):
    if images.dtype != torch.float:
        images = images.float()
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        features = inception_model(images).detach()
    return features.cpu().numpy()

def calculate_psnr(img1, img2):
    """
    Calculates the PSNR (Peak Signal-to-Noise Ratio) between two batches of images.
    Args:
        img1 (torch.Tensor): First batch of images (N, C, H, W).
        img2 (torch.Tensor): Second batch of images (N, C, H, W).
    Returns:
        float: Average PSNR value for the batch.
    """
    img1 = (img1.clamp(-1, 1) + 1) / 2
    img1 = (img1 * 255).type(torch.float16)
    
    max_pixel_value = max(img1.max(), img2.max())

    mse = F.mse_loss(img1, img2, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    
    psnr = 10 * torch.log10((max_pixel_value ** 2) / mse)
    
    return psnr.mean().item()


def calculate_fid(real_images, generated_images, device):
    """
    Calculates the FID (Fréchet Inception Distance) between real and generated images.
    Args:
        real_images (torch.Tensor): Batch of real images (N, C, H, W).
        generated_images (torch.Tensor): Batch of generated images (N, C, H, W).
        device (torch.device): Device to use for calculations.
    Returns:
        float: FID score.
    """
    inception = inception_v3(init_weights=True, transform_input=False).to(device)
    inception.eval()

    real_features = get_features(real_images, inception_model=inception)
    gen_features = get_features(generated_images, inception_model=inception)

    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid
