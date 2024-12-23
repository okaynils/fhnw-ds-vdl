import os

import hydra
from trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from data import NYUDepthV2, split_dataset
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Starting run: {cfg.run_name}")
    os.chdir(hydra.utils.get_original_cwd())
    logger.info(OmegaConf.to_yaml(cfg))
    
    

    train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size)

    class_weights_file = cfg.dataset.class_weights_file
    num_classes = cfg.model.num_classes
    class_weights = None

    if os.path.exists(class_weights_file):
        logger.info(f"Loading class weights from {class_weights_file}")
        class_weights = torch.load(class_weights_file, map_location=cfg.device)
    else:
        logger.info("Calculating class weights...")
        all_masks = [sample[1] for sample in train_dataset]

        flat_labels = np.concatenate([np.array(mask).flatten() for mask in all_masks])

        classes = list(range(0, 5))
        classes.append(255)

        class_weights = torch.tensor(
            compute_class_weight("balanced", classes=np.array(classes), y=flat_labels)[:-1], dtype=torch.float32, device=cfg.device
        )
        
        torch.save(class_weights, class_weights_file)
        logger.info(f"Class weights saved to {class_weights_file}")

    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    
    if cfg.criterion._target_ == 'torch.nn.CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)
    if cfg.criterion._target_ == 'core.FocalLoss':
        criterion = FocalLoss(gamma=cfg.criterion.gamma, weights=class_weights, ignore_index=255)

    logger.info(f'--- Model Configuration of {cfg.model._target_} ---')
    logger.info(model)
    logger.info(f'Total Parameters: {sum(p.numel() for p in model.parameters())}')
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=cfg.epochs,
        seed=cfg.seed,
        device=cfg.device,
        verbose=cfg.verbose,
        run_name=cfg.run_name,
        early_stopping_patience=cfg.trainer.early_stopping_patience,
        n_classes=cfg.model.num_classes
    )

    trainer.run(train_loader, val_loader)
    trainer.test(test_loader)

if __name__ == "__main__":
    main()
    