import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import os
import wandb
from core import EMA
import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, diffusion, optimizer, epochs, device, train_dataloader, val_dataloader=None, 
                 run_name='diffusion_model', project_name='diffusion_project', save_dir='models', ema_decay=0.995, 
                 sample_images_every=100, resolved_names=None, scheduler=None):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.run_name = run_name
        self.project_name = project_name
        self.save_dir = save_dir
        self.ema_decay = ema_decay
        self.resolved_names = resolved_names
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        wandb.init(project=self.project_name, name=self.run_name)
        self.run_id = wandb.run.id
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.sample_images_every = sample_images_every

        os.makedirs(self.save_dir, exist_ok=True)

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0.0

        for images, segments, depths, class_vectors, depth_vectors in self.train_dataloader:
            images = images.to(self.device)
            class_vectors = class_vectors.to(self.device)
            depth_vectors = depth_vectors.to(self.device)

            t = self.diffusion.sample_timesteps(images.size(0)).to(self.device)
            x_t, noise = self.diffusion.noise_images(images, t)

            predicted_noise = self.model(x_t, t, class_vectors, depth_vectors)

            loss = F.mse_loss(predicted_noise, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.ema.step_ema(self.ema_model, self.model)

            epoch_loss += loss.item()

        if self.scheduler:
            self.scheduler.step()

        avg_loss = epoch_loss / len(self.train_dataloader)
        return avg_loss

    def _validate_epoch(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, segments, depths, class_vectors, depth_vectors in self.val_dataloader:
                images = images.to(self.device)
                class_vectors = class_vectors.to(self.device)
                depth_vectors = depth_vectors.to(self.device)

                t = self.diffusion.sample_timesteps(images.size(0)).to(self.device)
                x_t, noise = self.diffusion.noise_images(images, t)

                predicted_noise = self.model(x_t, t, class_vectors, depth_vectors)
                loss = F.mse_loss(predicted_noise, noise)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_dataloader)
        return avg_val_loss

    def _save_model(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            model_name = f"{self.run_name}_{self.run_id}.pth"
            save_path = os.path.join(self.save_dir, model_name)
            torch.save(self.model.state_dict(), save_path)
            torch.save(self.ema_model.state_dict(), save_path.replace(".pth", "_ema.pth"))
            logger.info(f"Models saved to {self.save_dir} with val_loss {val_loss:.4f}")

    def _plot_samples(self, n=5, n_present_classes=3, depth_lower=0.1, depth_upper=3.0):
        indices = torch.stack([torch.randperm(18)[:n_present_classes] for _ in range(n)])
        class_vectors = torch.zeros((n, 18)).to(self.device)
        rows = torch.arange(n).unsqueeze(1)
        class_vectors[rows, indices] = 1
        
        depth_vectors = torch.rand(n, 18).to(self.device) * depth_upper + depth_lower
        
        class_labels = []
        for i in range(n):
            indices = torch.where(class_vectors[i] == 1)[0].tolist()
            labels = [(self.resolved_names[idx], depth_vectors[i, idx].item()) for idx in indices]
            class_labels.append(labels)

        default_sampled_images = self.diffusion.sample(self.model, n=n, class_vectors=class_vectors, depth_vectors=depth_vectors)
        ema_sampled_images = self.diffusion.sample(self.ema_model, n=n, class_vectors=class_vectors, depth_vectors=depth_vectors)
        
        fig_default, axes_default = plt.subplots(1, n, figsize=(n * 3, 3))
        for i, ax in enumerate(axes_default):
            img = default_sampled_images[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("\n".join([f"{cls}: {depth:.2f}" for cls, depth in class_labels[i]]), fontsize=8)
        fig_default.tight_layout()
        plt.close(fig_default)

        fig_ema, axes_ema = plt.subplots(1, n, figsize=(n * 3, 3))
        for i, ax in enumerate(axes_ema):
            img = ema_sampled_images[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.axis('off')
            ax.set_title("\n".join([f"{cls}: {depth:.2f}" for cls, depth in class_labels[i]]), fontsize=8)
        fig_ema.tight_layout()
        plt.close(fig_ema)
        
        wandb.log({"Default Model Samples": wandb.Image(fig_default),
                   "EMA Model Samples": wandb.Image(fig_ema)})

    def run(self):
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            current_lr = self.optimizer.param_groups[0]['lr']

            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()
            
            wandb.log({"epoch": epoch + 1,
                       "train_loss": train_loss,
                       "val_loss": val_loss,
                       "learning_rate": current_lr})
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if epoch % self.sample_images_every == 0:
                self._plot_samples()

            self._save_model(val_loss)

    def test(self, test_dataloader):
        self.model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for images, segments, depths, class_vectors, depth_vectors in test_dataloader:
                images = images.to(self.device)
                class_vectors = class_vectors.to(self.device)
                depth_vectors = depth_vectors.to(self.device)

                t = self.diffusion.sample_timesteps(images.size(0)).to(self.device)
                x_t, noise = self.diffusion.noise_images(images, t)

                predicted_noise = self.model(x_t, t, class_vectors, depth_vectors)
                loss = F.mse_loss(predicted_noise, noise)

                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_dataloader)
        logger.info(f"Test Loss: {avg_test_loss:.4f}")
        wandb.log({"test_loss": avg_test_loss})