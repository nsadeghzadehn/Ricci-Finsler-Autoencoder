# core/trainer_complete.py
# DATA-AWARE FINSLER AUTOENCODER TRAINER
# IID vs FLOW explicitly separated

import time
import torch
import torch.nn as nn
from core.finsler_ae import FinslerAE
from core.finsler_loss import compute_finsler_loss

class FinslerTrainer:
    def __init__(self, config):
        self.config = config

        # ------------------------------
        # DATA MODE
        # ------------------------------
        assert "data_mode" in config, "config['data_mode'] must be specified: 'iid' or 'flow'"
        assert config["data_mode"] in ["iid", "flow"]
        self.data_mode = config["data_mode"]

        # ------------------------------
        # Model
        # ------------------------------
        self.model = FinslerAE(
            input_dim=config["data_dim"],
            latent_dim=config["latent_dim"],
            diagonal_only=config.get("diagonal_only", True),
        )

        # ------------------------------
        # Optimization
        # ------------------------------
        self.beta_mode = config.get("beta_mode", "limited")
        self.beta_cap = config.get("beta_cap", 0.5)
        self.beta_schedule = config.get("beta_schedule", [1e-5, 1e-4, 1e-3])
        self.finsler_schedule = config.get("finsler_schedule", [1e-5, 1e-4, 1e-3])
        lr = config.get("lr", 5e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # ------------------------------
        # Ricci flow (optional)
        # ------------------------------
        self.use_ricci = config.get("use_ricci", False)
        if self.use_ricci:
            from core.ricci_flow import RicciFlow
            self.ricci_flow = RicciFlow(
                k_neighbors=config.get("ricci_k", 10),
                smoothing_factor=config.get("ricci_alpha", 0.15),
                iterations=config.get("ricci_iter", 2),
            )

        # ------------------------------
        # Logs
        # ------------------------------
        self.loss_history = []
        self.mse_history = []

    # ==================================================
    # Metrics (DATA-AWARE)
    # ==================================================
    def compute_trustworthiness(self, X, Z):
        if self.data_mode != "iid":
            return None
        try:
            from sklearn.manifold import trustworthiness
            k = min(12, max(3, X.shape[0] // 10))
            return float(trustworthiness(X.cpu().numpy(), Z.cpu().numpy(), n_neighbors=k))
        except Exception:
            return None

    def compute_dir_sim(self, X, Z):
        if self.data_mode != "flow":
            return None
        try:
            # compute trajectory tangents over the whole data
            h_x = X[1:] - X[:-1]
            h_x = torch.cat([h_x, h_x[-1:]], dim=0)
            h_x = h_x / (h_x.norm(dim=1, keepdim=True) + 1e-8)

            h_z = Z[1:] - Z[:-1]
            h_z = torch.cat([h_z, h_z[-1:]], dim=0)
            h_z = h_z / (h_z.norm(dim=1, keepdim=True) + 1e-8)

            # direction similarity: mean of pairwise inner products
            Sx = h_x @ h_x.T
            Sz = h_z @ h_z.T
            dir_sim = float((Sx * Sz).mean().item())
            return dir_sim
        except Exception:
            return None

    # ==================================================
    # Tangent definition (CORE)
    # ==================================================
    def compute_h_x(self, x):
        if self.data_mode == "iid":
            return torch.randn_like(x) * 1e-3
        if x.size(0) < 2:
            return torch.zeros_like(x)
        h = x[1:] - x[:-1]
        h = torch.cat([h, h[-1:]], dim=0)
        h = h / (h.norm(dim=1, keepdim=True) + 1e-8)
        return h

    # ==================================================
    # Train one epoch
    # ==================================================
    def train_epoch(self, loader, epoch):
        total_loss, total_mse, n = 0.0, 0.0, 0
        for batch in loader:
            x = batch[0].float()
            bsz = x.size(0)

            self.optimizer.zero_grad()
            out = self.model(x)

            # Reconstruction
            x_hat = out["x_hat"]
            mse = nn.functional.mse_loss(x_hat, x)
            loss = mse

            # Ricci smoothing
            if self.use_ricci:
                with torch.no_grad():
                    g_sm = self.ricci_flow.smooth_metrics(out["z"], out["g_diag"])
                    out["g_diag"] = g_sm
                    out["ginv_diag"] = 1.0 / (g_sm + 1e-8)

            # Finsler regularization
            if self.beta_mode != "zero":
                beta_scale = self.beta_schedule[min(epoch, len(self.beta_schedule)-1)]
                finsler_w = self.finsler_schedule[min(epoch, len(self.finsler_schedule)-1)]

                beta = out["beta"] * beta_scale
                if self.beta_mode == "limited":
                    beta = torch.clamp(beta, -self.beta_cap, self.beta_cap)

                h_x = self.compute_h_x(x)

                finsler = compute_finsler_loss(
                    h=h_x,
                    beta=beta,
                    g_diag=out.get("g_diag"),
                    ginv_diag=out.get("ginv_diag"),
                    beta_mode=self.beta_mode,
                    cap=self.beta_cap,
                )["total_loss"]

                loss = (1 - finsler_w) * mse + finsler_w * finsler

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * bsz
            total_mse += mse.item() * bsz
            n += bsz

        return total_loss/n, total_mse/n

    # ==================================================
    # Full training run
    # ==================================================
    def run_all(self, X):
        X = X.float()
        epochs = self.config.get("epochs", 10)
        batch_size = self.config.get("batch_size", 128)
        shuffle = self.data_mode == "iid"

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X),
            batch_size=batch_size,
            shuffle=shuffle,
        )

        print("\n" + "="*70)
        print(f"TRAINING FINSLER AE | data_mode = {self.data_mode.upper()}")
        print("="*70)

        start = time.time()
        for ep in range(epochs):
            loss, mse = self.train_epoch(loader, ep)
            self.loss_history.append(loss)
            self.mse_history.append(mse)
            if ep==0 or ep==epochs-1 or (ep+1)%3==0:
                print(f"Epoch {ep+1:3d}/{epochs} | Loss {loss:.4f} | MSE {mse:.4f}")

        total_time = time.time() - start

        # Evaluation
        with torch.no_grad():
            Z = self.model.encoder(X)
            X_hat = self.model.decoder(Z)
            final_mse = nn.functional.mse_loss(X_hat, X).item()
            trust = self.compute_trustworthiness(X, Z)
            dir_sim = self.compute_dir_sim(X, Z)

        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)

        msg = f"Final MSE: {final_mse:.4f}"
        if trust is not None:
            msg += f" | Trust: {trust:.4f}"
        if dir_sim is not None:
            msg += f" | DirSim: {dir_sim:.4f}"
        print(msg)

        return {
            "final": {
                "mse": final_mse,
                "trust": trust,
                "dir_sim": dir_sim,
                "loss": self.loss_history[-1],
                "time": total_time,
            },
            "latents": Z,
            "loss_history": self.loss_history,
            "mse_history": self.mse_history,
        }
