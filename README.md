# Advanced Deep Learning (vdl) Mini Challenge: DDPM + Conditional Diffuison on NYU Depth v2

This repository contains a PyTorch-based framework for training diffusion models, based on the original DDPM paper by Ho et al ([2020](https://arxiv.org/pdf/2006.11239)), that generate images **conditioned** on both semantic *class vectors* and *depth vectors*. The code is centered around the [**NYU Depth V2**](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) dataset by Silberman et al ([2012](https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf)), where each sample includes an RGB image, a segmentation mask, and a depth map. The model takes in class vectors and depth vectors, then predicts the noise in a diffusion framework to reconstruct or generate new images.

I have built this project during the fall semester of 2025 in the B.S. Data Science program at FHNW in the Advanced Deep Learning course (vdl).

Below you will find an overview of the repository structure, how to set up and run the code, and how to customize experiments using Hydra.

---

## Table of Contents
1. [Key Features](#key-features)  
2. [Repository Structure](#repository-structure)  
3. [Installation & Requirements](#installation--requirements)  
4. [Quick Start](#quick-start)  
5. [Training and Validation](#training-and-validation)  
6. [Evaluating Metrics](#evaluating-metrics)  
7. [Hydra Configuration](#hydra-configuration)  
8. [Project Workflow](#project-workflow)  
9. [Acknowledgements](#acknowledgements)

---

## Key Features
- **Diffusion Pipeline**: Implements forward noising (`noise_images`) and reverse sampling (`sample`) for image synthesis.  
- **UNet Architectures**: Three main variants:
  1. **UNet** (with optional dropout)  
  2. **UNet_Attn** (with self-attention blocks)  
  3. **UNet_Baseline** (simplified structure)  
- **Class & Depth Conditioning**: The model can be conditioned on segmentation classes and their associated depth information.  
- **Exponential Moving Average (EMA)**: Maintains a shadow copy of the model that stabilizes training and often yields better samples.  
- **W&B Integration**: Logs training/validation metrics, sample images, FID & PSNR metrics to Weights & Biases.  
- **Hydra Configurations**: A flexible system to configure experiments (model, trainer, dataset, etc.) via YAML files.  
- **NYU Depth V2 Integration**: Automatic download and data pre-processing (histogram equalization, resizing, etc.).  
- **Metrics**: FID and PSNR are computed for quantitative evaluation.  

---

## Repository Structure
```
.
├── analyzer.py        # Tools for analyzing training runs, plotting metrics, sampling images
├── diffusion.py       # Core Diffusion class handling forward noise & reverse sampling
├── metrics.py         # FID and PSNR computation
├── train.py           # Entry point for training with Hydra
├── trainer.py         # Trainer class orchestrating the training/validation loop
├── utils.py           # Helper function for unnormalizing images
│
├── core/
│   ├── modules.py        # Core building blocks (DoubleConv, Down, Up, SelfAttention, etc.)
│   ├── unet.py           # UNet model variant
│   ├── unet_attn.py      # UNet model with self-attention
│   ├── unet_baseline.py  # UNet baseline model
│   └── utils.py          # EMA (Exponential Moving Average) class and other utilities
│
├── data/
│   ├── nyuv2.py          # NYU Depth V2 dataset class
│   ├── utils.py          # Dataset splitting, histogram equalization, unnormalization
│   └── __init__.py
│
├── configs/
│   ├── dataset/          # Dataset-specific configs (nyuv2.yaml)
│   ├── model/            # Model-specific configs
│   ├── optimizer/        # Optimizer configs
│   ├── experiment/       # Predefined experiment recipes (dropout, attention, overfit, etc.)
│   ├── trainer/          # Trainer configs (epochs, batch size, diffusion settings, etc.)
│   ├── config.yaml       # Base config
│   └── ...
│
├── scripts/
│   ├── download.sh       # Syncs models from remote server
│   ├── submit_job.sh     # Slurm job submission script
│   └── upload.sh         # Syncs code to remote server
│
└── requirements.txt / environment.yml (not shown, assume your usual environment file)
```

---

## Installation & Requirements

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up Python environment** (conda, virtualenv, etc.). Example using conda:

```bash
conda create -n diffusion python=3.9 -y
conda activate diffusion
```

3. **Install dependencies:**
  - If using `requirements.txt`, run:
    ```bash
    pip install -r requirements.txt
    ```
  - Otherwise, refer to your environment file (`environment.yml`) or install packages manually.

4. *(Optional)* **Weights & Biases**:
  - Create a free account at Weights & Biases if you want to track experiments.
  - Run `wandb login` in your terminal to authenticate.

## Quick Start
Below is a high-level guide for training a model end-to-end:

1. Configure your run parameters in a Hydra config file (e.g., `configs/experiment/unet.yaml` or any `experiment` config).

2. Run the training script:
  ```bash
  python train.py experiment=unet_attn
  ```
  - Hydra merges `configs/config.yaml` with `configs/experiment/unet_attn.yaml`.
  - This will train a UNet with attention for 500 epochs, as defined in that config.

3. Monitor logs:
  - If using Weights & Biases, logs will appear in your W&B project automatically.
  - Alternatively, logs will print to your console or be stored in local `.out`/`.err` files (if using Slurm).

## Training and Validation
### How It Works
- **Diffusion**: In each training step, an image is noised at a random timestep *t*. The model attempts to predict the injected noise.
- **Conditioning**: The model receives additional embeddings for each semantic class and an average depth value of that class within the image.
- **Trainer**:
  - **Train**: Minimizes MSE loss between predicted noise and actual noise.
  - **Validate**: Evaluates MSE on the validation set and checks FID/PSNR (periodically).
  - **EMA**: Maintains an exponential moving average copy of the model, which is also periodically evaluated.

### Commands
Run a default experiment:
```bash
python train.py
```

Run a custom experiment, e.g., UNet with self-attention:

```bash
python train.py experiment=unet_attn
```

Override hyperparameters inline (example changes learning rate and number of epochs):

```bash
python train.py experiment=unet_attn optimizer.lr=1e-5 trainer.epochs=100
```

## Evaluating Metrics
### FID (Fréchet Inception Distance)
- Evaluates the quality and diversity of generated samples compared to real samples.
- Implemented in `metrics.py` using PyTorch and SciPy for matrix operations.

### PSNR (Peak Signal-to-Noise Ratio)
- Measures the reconstruction quality of generated images (e.g., how close they are to ground truth).
- Also implemented in `metrics.py`.

During validation, these metrics are computed and automatically logged to W&B under the keys `"FID"` and `"PSNR"`.

## Hydra Configuration
Hydra manages the configuration via YAML files inside `configs/`:
- `config.yaml`: Base config containing default seeds and device info.
- `dataset/*.yaml`: Dataset-specific configs (like `nyuv2.yaml`).
- `model/*.yaml`: Model definitions (like `unet.yaml`, `unet_attn.yaml`).
- `trainer/*.yaml`: Trainer defaults, including epochs, batch size, diffusion hyperparams, etc.
- `experiment/*.yaml`: Ready-made experiment recipes combining different configs.

### Usage
```bash
python train.py experiment=<experiment_name>
```

For example:

```bash
python train.py experiment=unet_attn_dropout_0.3
```

Hydra merges:

1. `configs/config.yaml`
2. `configs/experiment/unet_attn_dropout_0.3.yaml`
3. The relevant `model`, `optimizer`, and `trainer` default YAMLs.

You can override any parameter inline:

```bash
python train.py \
  experiment=unet_attn_dropout_0.3 \
  trainer.epochs=2000 \
  optimizer.weight_decay=1e-4
```