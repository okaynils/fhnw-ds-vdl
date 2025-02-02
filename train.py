import os
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging

from trainer import Trainer
from data.nyuv2 import NYUDepthV2
from data.utils import split_dataset, HistogramEqualization
from torchvision import transforms
from torch.utils.data import DataLoader
from diffusion import Diffusion

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training script that gets called by Hydra.
    Usage example:
      python train.py trainer.epochs=200 optimizer.lr=2e-5
    """

    logger.info("=== CONFIGS ===")
    logger.info(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)

    mean = cfg.dataset.mean
    std = cfg.dataset.std

    image_t = transforms.Compose([
        transforms.CenterCrop(400),
        HistogramEqualization() if cfg.dataset.histogram_eq else transforms.Lambda(lambda x: x),
        transforms.Resize(cfg.trainer.diffusion.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    seg_t = transforms.Compose([
        transforms.CenterCrop(400),
        transforms.Resize(cfg.trainer.diffusion.img_size),
        transforms.ToTensor()
    ])

    depth_t = transforms.Compose([
        transforms.CenterCrop(400),
        transforms.Resize(cfg.trainer.diffusion.img_size),
        transforms.ToTensor()
    ])

    dataset = NYUDepthV2(
        root=cfg.dataset.root,
        download=cfg.dataset.download,
        preload=cfg.dataset.preload,
        image_transform=image_t,
        seg_transform=seg_t,
        depth_transform=depth_t,
        n_classes=cfg.dataset.n_classes,
        filtered_classes=cfg.dataset.filtered_classes
    )
    
    train_ratio = cfg.dataset.train_ratio
    val_ratio = cfg.dataset.val_ratio
    test_ratio = cfg.dataset.test_ratio
    
    if cfg.dataset.overfit_test == True:
        logger.info("Dataset overfitting mode enabled. Using only 24 samples.")
      
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset[:24], 
            train_ratio=train_ratio, 
            val_ratio=val_ratio, 
            test_ratio=test_ratio,
            random_seed=cfg.seed
        )
    else:
        train_dataset, val_dataset, test_dataset = split_dataset(
            dataset,
            train_ratio=train_ratio, 
            val_ratio=val_ratio, 
            test_ratio=test_ratio,
            random_seed=cfg.seed
        )

    train_loader = DataLoader(train_dataset, batch_size=cfg.trainer.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.trainer.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg.trainer.batch_size, shuffle=False)

    model = instantiate(cfg.model)
    model.to(cfg.device)

    logger.info("=== MODEL ===")
    logger.info(model)

    diffusion_params = cfg.trainer.diffusion
    diffusion = Diffusion(
        noise_steps=diffusion_params.noise_steps,
        beta_start=diffusion_params.beta_start,
        beta_end=diffusion_params.beta_end,
        img_size=diffusion_params.img_size,
        device=diffusion_params.device
    )

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = None
    if cfg.trainer.scheduler is not None:
        scheduler = instantiate(cfg.trainer.scheduler, optimizer=optimizer)

    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.trainer.epochs,
        device=cfg.device,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        run_name=cfg.trainer.run_name,
        project_name=cfg.trainer.project_name,
        save_dir=cfg.trainer.save_dir,
        ema_decay=cfg.trainer.ema_decay,
        sample_images_every=cfg.trainer.sample_images_every,
        resolved_names=getattr(dataset, "resolved_names", None),
        rgb_mean=mean,
        rgb_std=std
    )

    trainer.run()

    trainer.test(test_loader)


if __name__ == "__main__":
    main()
