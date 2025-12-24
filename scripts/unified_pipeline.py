# scripts/unified_pipeline.py
# UNIFIED PIPELINE – DATA-AWARE FINSLER AUTOENCODER

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import os
import torch
import numpy as np
import json
from core.trainer_complete import FinslerTrainer

# -------------------------------
# Load datasets
# -------------------------------
def load_datasets():
    datasets = {}

    # IID synthetic data
    X_syn = np.random.randn(1000, 50).astype(np.float32)
    datasets["synthetic"] = {
        "X": torch.tensor(X_syn),
        "data_mode": "iid"
    }

    # FLOW data (ordered trajectory – DO NOT SHUFFLE)
    t = np.linspace(0, 20, 5000)
    X_flow = np.sin(t).reshape(-1, 1)
    X_flow = np.tile(X_flow, (1, 50)).astype(np.float32)
    datasets["flow"] = {
        "X": torch.tensor(X_flow),
        "data_mode": "flow"
    }

    return datasets

# -------------------------------
# Ricci sweep (IID only)
# -------------------------------
def run_ricci_sweep(X, output_dir):
    ricci_params = [(5,1),(5,2),(5,4),(10,1),(10,2),(10,4),(20,1),(20,2),(20,4)]
    results = []

    for k, it in ricci_params:
        config = {
            "data_dim": X.shape[1],
            "latent_dim": 8,
            "epochs": 5,
            "batch_size": 128,
            "use_ricci": True,
            "ricci_k": k,
            "ricci_iter": it,
            "data_mode": "iid"
        }

        trainer = FinslerTrainer(config)
        out = trainer.run_all(X)

        results.append({
            "ricci_k": k,
            "ricci_iter": it,
            "loss": out["final"]["loss"],
            "mse": out["final"]["mse"],
            "trust": out["final"]["trust"]
        })

    print("\nRicci sweep completed")
    return results

# -------------------------------
# Multi-seed experiments (IID + FLOW)
# -------------------------------
def run_multi_seed_experiments(datasets, output_dir, seeds=[42, 123, 999]):
    results = {}

    for name, pack in datasets.items():
        X = pack["X"]
        data_mode = pack["data_mode"]

        all_mse, all_dirsim = [], []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            config = {
                "data_dim": X.shape[1],
                "latent_dim": 8,
                "epochs": 5,
                "batch_size": 128,
                "beta_mode": "limited",
                "data_mode": data_mode
            }

            trainer = FinslerTrainer(config)
            out = trainer.run_all(X)

            all_mse.append(out["final"]["mse"])
            ds_val = out["final"].get("dir_sim", 0.0)
            all_dirsim.append(ds_val if ds_val is not None else 0.0)

        results[name] = {
            "MSE_mean": float(np.mean(all_mse)),
            "MSE_std": float(np.std(all_mse)),
            "DirSim_mean": float(np.mean(all_dirsim)),
            "DirSim_std": float(np.std(all_dirsim))
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "multi_seed_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    return results

# -------------------------------
# Main
# -------------------------------
def main():
    output_dir = "outputs"
    datasets = load_datasets()

    print("="*70)
    print("UNIFIED PIPELINE – DATA-AWARE FINSLER AE")
    print("="*70)

    print("\n[1] Ricci sweep (IID synthetic)")
    ricci_results = run_ricci_sweep(datasets["synthetic"]["X"], output_dir)
    print(ricci_results)

    print("\n[2] Multi-seed experiments (IID + FLOW)")
    multi_seed_results = run_multi_seed_experiments(datasets, output_dir)
    print(json.dumps(multi_seed_results, indent=2))

    print("\nAll experiments completed. Outputs saved to:", output_dir)

if __name__ == "__main__":
    main()
