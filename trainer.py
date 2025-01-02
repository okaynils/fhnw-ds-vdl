import torch
import torch.nn.functional as F
import copy
import os
import wandb
from core import EMA
import logging
from metrics import calculate_fid, calculate_psnr
import matplotlib.pyplot as plt
from data import unnormalize

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, diffusion, optimizer, epochs, device, train_dataloader, val_dataloader=None, 
                 run_name='diffusion_model', project_name='diffusion_project', save_dir='models', ema_decay=0.995, 
                 sample_images_every=100, resolved_names=None, scheduler=None, rgb_mean=[.5, .5, .5], rgb_std=[.5, .5, .5]):
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

        # Initialize EMA
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        # Initialize Weights & Biases
        wandb.init(project=self.project_name, name=self.run_name)
        self.run_id = wandb.run.id
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.sample_images_every = sample_images_every
        
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        # Create a models directory if it doesn't exist
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
            ema_model_name = f"{self.run_name}_ema_{self.run_id}.pth"
            ema_save_path = os.path.join(self.save_dir, ema_model_name)
            torch.save(self.ema_model.state_dict(), ema_save_path)
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

    def _validate_samples(self, num_samples=4, n_plot=5):
        """
        Validates the model by computing FID and PSNR and logs the results.
        Args:
            n_plot (int): Number of real and generated samples to log on WandB.
        """
        self.model.eval()
        real_images_list = []
        generated_images_list = []
        ema_generated_images_list = []
        psnr_values = []
        plot_real_images = []
        plot_generated_images = []
        plot_ema_generated_images = []

        with torch.no_grad():
            for idx, (images, _, _, class_vectors, depth_vectors) in enumerate(self.val_dataloader):
                images = images.to(self.device)
                class_vectors = class_vectors.to(self.device)
                depth_vectors = depth_vectors.to(self.device)

                # Collect real images
                real_images_list.append(images)
                if len(plot_real_images) < n_plot:
                    plot_real_images.append(images[:n_plot])

                # Generate corresponding images using the default model
                generated_images = self.diffusion.sample(self.model, n=images.size(0),
                                                          class_vectors=class_vectors,
                                                          depth_vectors=depth_vectors)
                generated_images_list.append(generated_images)
                if len(plot_generated_images) < n_plot:
                    plot_generated_images.append(generated_images[:n_plot])

                # Generate corresponding images using the EMA model
                ema_generated_images = self.diffusion.sample(self.ema_model, n=images.size(0),
                                                             class_vectors=class_vectors,
                                                             depth_vectors=depth_vectors)
                ema_generated_images_list.append(ema_generated_images)
                if len(plot_ema_generated_images) < n_plot:
                    plot_ema_generated_images.append(ema_generated_images[:n_plot])

                # Compute PSNR for the batch
                batch_psnr = calculate_psnr(images, generated_images)
                psnr_values.append(batch_psnr)

                if len(real_images_list) * images.size(0) >= num_samples:
                    break

        # Concatenate collected images
        real_images = torch.cat(real_images_list, dim=0)[:num_samples]
        generated_images = torch.cat(generated_images_list, dim=0)[:num_samples]
        ema_generated_images = torch.cat(ema_generated_images_list, dim=0)[:num_samples]

        # Compute FID
        fid_score = calculate_fid(real_images, generated_images, self.device)

        # Average PSNR
        avg_psnr = sum(psnr_values) / len(psnr_values)

        # Log metrics to WandB
        wandb.log({"FID": fid_score, "PSNR": avg_psnr})
        logger.info(f"FID: {fid_score:.4f}, PSNR: {avg_psnr:.4f}")

        # Log sample images to WandB
        self._log_samples_to_wandb(torch.cat(plot_real_images, dim=0)[:n_plot],
                                   torch.cat(plot_generated_images, dim=0)[:n_plot],
                                   torch.cat(plot_ema_generated_images, dim=0)[:n_plot],
                                   class_vectors[:n_plot], depth_vectors[:n_plot])

    def _log_samples_to_wandb(self, real_images, generated_images, ema_generated_images, class_vectors, depth_vectors):
        """
        Logs real and generated image samples to WandB.
        Args:
            real_images (torch.Tensor): Batch of real images to log.
            generated_images (torch.Tensor): Batch of generated images to log.
            ema_generated_images (torch.Tensor): Batch of EMA-generated images to log.
            class_vectors (torch.Tensor): Corresponding class vectors.
            depth_vectors (torch.Tensor): Corresponding depth vectors.
        """
        fig, axes = plt.subplots(3, len(real_images), figsize=(len(real_images) * 3, 9))

        real_images = unnormalize(real_images.cpu(), mean=self.rgb_mean, std=self.rgb_std)
        generated_images = unnormalize(generated_images.cpu(), mean=self.rgb_mean, std=self.rgb_std)
        ema_generated_images = unnormalize(ema_generated_images.cpu(), mean=self.rgb_mean, std=self.rgb_std)
        
        # Add row labels on the left
        row_labels = ["Real Images", "Generated Images", "Generated Images (EMA)"]
        for row_idx, label in enumerate(row_labels):
            axes[row_idx][0].set_ylabel(label, fontsize=12, rotation=0, labelpad=50, ha='right', va='center')

        for i, ax in enumerate(axes[0]):
            img = real_images[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        for i, ax in enumerate(axes[1]):
            img = generated_images[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        for i, ax in enumerate(axes[2]):
            img = ema_generated_images[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            labels = [f"{self.resolved_names[idx]}: {depth_vectors[i, idx]:.2f}"
                    for idx in torch.where(class_vectors[i] == 1)[0].tolist()]
            ax.set_xlabel("\n".join(labels), fontsize=8)

        # Adjust layout to make space for row labels and x-labels
        fig.subplots_adjust(hspace=0.5, wspace=0.3)

        fig.tight_layout()
        wandb.log({"Validation Samples": wandb.Image(fig)})
        plt.close(fig)


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
                self._validate_samples(num_samples=self.val_dataloader.batch_size, n_plot=10)

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