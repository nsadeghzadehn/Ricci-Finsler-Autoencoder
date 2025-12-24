# Ricci-driven Finsler Autoencoder (F-AE)

This repository provides the official implementation of the **Ricci-driven Finsler Autoencoder (F-AE)**, a geometry-aware autoencoder designed to learn **direction-sensitive latent representations**. The proposed method integrates a Randers-type Finsler loss with discrete Ricci-flow smoothing to capture anisotropic structures that are not addressed by standard isotropic or Riemannian autoencoders.

The code accompanies the manuscript submitted to a Q1 journal and enables **full reproducibility of all reported experimental results**, including quantitative tables and figures.

---

## Key Contributions

- A Randers-type Finsler loss for direction-sensitive representation learning.
- Discrete Ricci-flow smoothing applied to latent geometry.
- A unified evaluation pipeline supporting multiple anisotropic datasets.
- Fully reproducible experiments with fixed random seeds.

---

## Repository Structure

- `core/`  
  Core model components, including the Finsler autoencoder, loss functions, Ricci-flow smoothing, and the complete training routine.

- `scripts/`  
  Executable pipelines for training, evaluation, and result generation.  
  `unified_pipeline.py` serves as the main entry point for all experiments.

- `datasets/`  
  Synthetic and flow-based datasets used in the experiments.

- `outputs/`  
  Serialized experimental results, including multi-seed statistics and Ricci parameter sweeps.

- `figures/`  
  Generated figures corresponding directly to those reported in the manuscript.

- `tables/`  
  LaTeX-formatted tables reproducing the quantitative results in the paper.

- `Manuscript/`  
  LaTeX source files of the associated manuscript.

---

## Installation

1. Make sure Python 3.8 or higher is installed.
2. Install dependencies:
```bash
pip install -r requirements.txt
