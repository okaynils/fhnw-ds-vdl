import copy
import os
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt

from diffusion import Diffusion
from data.utils import unnormalize

class Analyzer:
    def __init__(self, model: nn.Module, device: str="cpu", project_name: str="vdl", entity_name: str="okaynils", diffusion_params: dict=None):
        self.model = model
        self.ema_model = None
        self.device = device
        self.project_name = project_name
        self.entity_name = entity_name
        self.history = None
        self.run_name = None
        self.elapsed_time = None
        
        if diffusion_params == None:
            diffusion_params = OmegaConf.load('configs/trainer/default.yaml').diffusion
            
        self.diffusion = Diffusion(
            noise_steps=diffusion_params.noise_steps,
            beta_start=diffusion_params.beta_start,
            beta_end=diffusion_params.beta_end,
            img_size=diffusion_params.img_size,
            device=self.device
        )
    
    def model_receipt(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        device = next(self.model.parameters()).device
        
        print(f"--- Model Receipt for {self.model.__class__.__name__} ---")
        if self.elapsed_time:
            print(f"\nTraining Time: {self.elapsed_time/60**2:.2f} hours")
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {non_trainable_params}")
        print(f"Device: {device}")
        
        print("\nModel Architecture:\n")
        print(self.model)
        
    def plot(self, run_id: str):
        self._fetch_data(run_id)
        
        train_loss = [entry['train_loss'] for entry in self.history if entry['train_loss'] is not None]
        val_loss = [entry['val_loss'] for entry in self.history if entry['val_loss'] is not None]
        fid = [entry['FID'] for entry in self.history if entry['FID'] is not None]
        psnr = [entry['PSNR'] for entry in self.history if entry['PSNR'] is not None]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        axes[0].plot(train_loss, label='Train Loss', zorder=2)
        axes[0].plot(val_loss, label='Validation Loss', zorder=1)
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].legend()
        axes[0].grid(True)

        fid_epochs = [i * 50 for i in range(len(fid))]
        axes[1].plot(fid_epochs, fid, label='FID', marker='o', color='orange')
        axes[1].set_title('FID Metric')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('FID Value')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True)

        psnr_epochs = [i * 50 for i in range(len(psnr))]
        axes[2].plot(psnr_epochs, psnr, label='PSNR', marker='x', color='green')
        axes[2].set_title('PSNR Metric')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('PSNR Value')
        axes[2].legend()
        axes[2].grid(True)

        fig.suptitle(f'Run: {self.run_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
    def sample_images(self, run_id: str, resolved_names, class_vects: torch.tensor, depth_vects: torch.tensor, n_samples: int = 5):
        self._load_model_weights(run_id)

        dataset_config = OmegaConf.load('configs/dataset/nyuv2.yaml')
        mean, std = dataset_config.mean, dataset_config.std

        if len(class_vects) != n_samples:
            class_vects = class_vects.repeat(n_samples, 1)
        if len(depth_vects) != n_samples:
            depth_vects = depth_vects.repeat(n_samples, 1)

        x = torch.randn((n_samples, 3, 64, 64)).to(self.device)

        default_sampled_images = self.diffusion.sample(model=self.model, n=n_samples, x=x, class_vectors=class_vects.to(self.device), depth_vectors=depth_vects.to(self.device))
        ema_sampled_images = self.diffusion.sample(model=self.ema_model, n=n_samples, x=x, class_vectors=class_vects.to(self.device), depth_vectors=depth_vects.to(self.device))

        def display_images(images, title, class_vects, depth_vects):
            fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
            fig.suptitle(title, fontsize=16)

            for i in range(n_samples):
                ax = axes[i]
                ax.imshow(unnormalize(images[i].cpu(), mean, std).permute(1, 2, 0))

                ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True)
                ax.set_xticks([])
                ax.set_yticks([])

                present_classes = [resolved_names[j] for j in range(len(class_vects[i])) if class_vects[i][j] == 1]
                depths = [depth_vects[i][j].item() for j in range(len(class_vects[i])) if class_vects[i][j] == 1]

                label_text = "\n".join([f"{cls}: {depth:.2f}m" for cls, depth in zip(present_classes, depths)])
                ax.set_xlabel(label_text, fontsize=8)

            plt.subplots_adjust(bottom=0.25, top=0.85, wspace=0.3)
            plt.show()

        display_images(default_sampled_images, "Default Model Samples", class_vects, depth_vects)
        display_images(ema_sampled_images, "EMA Model Samples", class_vects, depth_vects)

    def _get_model_path(self, run_id: str):
        print(f"\nSearching for model weights for run {run_id}...")
        models = os.listdir('models')
        model_path = None
        ema_model_path = None
        for model in models:
            run_name = model.split('_')
            if len(run_name) > 1:
                if model.split('_')[-2] == 'ema' and run_name[-1].split('.')[0] == run_id:
                    ema_model_path = model
                    print(f'Found EMA model: {ema_model_path}!')
                    continue
                if run_name[-1].split('.')[0] == run_id:
                    model_path = model
                    print(f'Found default model: {model_path}!')
        return model_path, ema_model_path
        
    def _load_model_weights(self, run_id: str):
        model_path, ema_model_path = self._get_model_path(run_id)
        
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(f'models/{model_path}', map_location=self.device, weights_only=True))
        self.model.to(self.device)
        if ema_model_path:
            self.ema_model.load_state_dict(torch.load(f'models/{ema_model_path}', map_location=self.device, weights_only=True))
            self.model.to(self.device)
    
    def _fetch_data(self, run_id: str):
        api = wandb.Api()
        try:
            run = api.run(f"{self.entity_name}/{self.project_name}/{run_id}")
            self.run_name = run.name
            self.elapsed_time = run.summary.get('_runtime', None)
            self.history = run.scan_history()
        except wandb.errors.CommError as e:
            raise ValueError(f"Error fetching run: {e}")
