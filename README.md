# Stable Diffusion (v1.5) Implementation from Scratch

This repository contains a PyTorch implementation of the **Stable Diffusion (v1.5)** architecture built from scratch. The project focuses on **Text-to-Image** generation using DDPM!

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
  - [1. Variational Autoencoder (VAE)](#1-variational-autoencoder-vae)
  - [2. Text Conditioning via CLIP](#2-text-conditioning-via-clip)
  - [3. Conditional U-Net Denoising Network](#3-conditional-u-net-denoising-network)
  - [4. VAE Decoder Reconstruction](#4-vae-decoder-reconstruction)
- [DDPM Mathematical Formulation](#ddpm-mathematical-formulation)
- [Pre-trained Weights & Models Used](#pre-trained-weights--models-used)
- [Usage](#usage)
- [Output Demos](#output-demos)

---

## Architecture Overview

Generating high-resolution images directly in pixel space is computationally expensive. Stable Diffusion addresses this by operating inside a lower-dimensional **latent space** learned by a Variational Autoencoder (VAE).

```text
+-------------------+
|    Text Prompt    |
+---------+---------+
          |
          v
+-------------------+        +----------------------------+
|   CLIP Tokenizer  |------->| Conditional Text Embeddings| (77, 768)
|   & Text Encoder  |        +-------------+--------------+
+-------------------+                      |
                                           | Cross-Attention
                                           v
+-------------------+        +----------------------------+        +----------------------+
| Pure Latent Noise |------->|    Time-Conditioned U-Net  |------->| Final Denoised Latent|
|   x_T ~ N(0, I)   |        |       Denoiser Loop        |        |    x_0 (4x64x64)     |
+-------------------+        +----------------------------+        +----------+-----------+
                                                                              |
                                                                              v
                                                                   +----------------------+
                                                                   |     VAE Decoder      |
                                                                   +----------+-----------+
                                                                              |
                                                                              v
                                                                   +----------------------+
                                                                   |  Final Output Image  |
                                                                   |     (3x512x512)      |
                                                                   +----------------------+
```

### 1. Variational Autoencoder (VAE)

- **Compression:** Encodes an RGB pixel image $I \in \mathbb{R}^{3 \times 512 \times 512}$ into a lower-dimensional latent representation $z \in \mathbb{R}^{4 \times 64 \times 64}$.
- **Spatial Reduction:** $8\times$ spatial downsampling reduces computational overhead significantly while preserving essential visual features and structural context.

### 2. Text Conditioning via CLIP

- **Tokenization:** Text prompts are padded or truncated to a fixed context length of $77$ tokens.
- **Embeddings:** A pre-trained CLIP text encoder converts token IDs into text prompt embeddings $c \in \mathbb{R}^{77 \times 768}$.
- **Injection:** Prompt embeddings guide the U-Net denoiser at each sampling step using cross-attention mechanisms.

### 3. Conditional U-Net Denoising Network

- **Role:** Predicts the noise present at timestep $t$ in a latent state $x_t$.
- **Structure:** Built using ResNet blocks, downsampling/upsampling layers, skip connections, and attention blocks.
- **Cross-Attention:** Blends text embeddings $c$ into intermediate feature maps:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### 4. VAE Decoder Reconstruction

- **Reconstruction:** After $T$ iterative denoising steps yield clean latent $x_0 \in \mathbb{R}^{4 \times 64 \times 64}$, the VAE Decoder expands the latent representation back to pixel space $I_\text{generated} \in \mathbb{R}^{3 \times 512 \times 512}$.

---

## DDPM Mathematical Formulation

This project follows the core equations from the paper *Denoising Diffusion Probabilistic Models*:

**Forward Noising Process:** Adds Gaussian noise to clean latent $x_0$ at any arbitrary timestep $t$:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

**Estimating Clean Latent $x_0$:** Reconstructs the estimated target image $x_0$ from noise prediction $\epsilon_\theta$:

$$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t, c) \right)$$

**Reverse Step Denoising (Posterior Mean):** Computes the previous step latent $x_{t-1}$ from current state $x_t$:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right) + \sigma_t z, \quad \text{where } z \sim \mathcal{N}(0, \mathbf{I})$$

**Classifier-Free Guidance (CFG):** Combines conditioned and unconditioned noise predictions with guidance scale $s$ to enforce prompt alignment:

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot \left( \epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset) \right)$$

---

## Pre-trained Weights & Models Used

This repository uses pre-trained checkpoint weights and tokenizer files from Hugging Face:

| Component | Model Name / File | Source Link |
|---|---|---|
| Stable Diffusion Checkpoint | `v1-5-pruned-emaonly.ckpt` | RunwayML Hugging Face Hub |
| CLIP Tokenizer | `openai/clip-vit-large-patch14` | OpenAI Hugging Face Hub |
| Tokenizer Configs | `vocab.json` & `merges.txt` | Stable Diffusion v1.5 Repository |

---

## Usage

To generate images, open and run the included notebook:

1. 📄 `sd/demo.ipynb`
2. Activate your Python environment (or run on Google Colab GPU).
3. Open `sd/demo.ipynb`.
4. Execute the cells to initialize the models and run text-to-image generation.

---

## Output Demos

### Demo 1

**Prompt:** "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution"

![Demo 1 Output](/images/prompt1_output.png)

### Demo 2

**Prompt:** "An ancient Japanese temple surrounded by cherry blossom trees during sunset, watercolor painting style, high detail"

![Demo 2 Output](/images/prompt2_output.png)

### Demo 3

**Prompt:** "A cup of coffee on a table"

![Demo 3 Output](/images/prompt3_output.png)
