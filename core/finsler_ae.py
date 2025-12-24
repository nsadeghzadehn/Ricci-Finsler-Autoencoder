import torch
import torch.nn as nn

# ------------------------------
# BetaNet
# ------------------------------
class BetaNet(nn.Module):
    """شبکه برای تولید بردار β"""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# ------------------------------
# GNet
# ------------------------------
class GNet(nn.Module):
    """شبکه برای تولید ماتریس G ^(مورب^)"""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z):
        g = self.net(z)
        return torch.exp(g) + 1e-6

# ------------------------------
# FinslerAE
# ------------------------------
class FinslerAE(nn.Module):
    """مدل کامل AutoEncoder فینسلر"""
    def __init__(self, input_dim, latent_dim, diagonal_only=True):
        super().__init__()
        self.diagonal_only = diagonal_only

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

        self.beta_net = BetaNet(latent_dim, input_dim)
        self.g_net = GNet(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        beta = self.beta_net(z) * 0.01
        g_diag = self.g_net(z)
        ginv_diag = 1.0 / g_diag

        return {
            'x_hat': x_hat,
            'z': z,
            'beta': beta,
            'g_diag': g_diag,
            'ginv_diag': ginv_diag
        }

    # ------------------------------
    # V_theta – pushforward latent perturbation امن
    # ------------------------------
    def V_theta(self, z, x):
        """
        Compute safe latent perturbation direction.
        ابعاد: [batch, latent_dim]
        """
        # detach برای جلوگیری از gradient اضافی
        z = z.detach()
        z.requires_grad_(True)

        # forward decoder
        x_hat = self.decoder(z)

        # perturbation direction امن: gradient sum(x_hat) w.r.t z
        delta_z = torch.autograd.grad(
            outputs=x_hat.sum(),
            inputs=z,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # normalize
        delta_z = delta_z / (delta_z.norm(dim=1, keepdim=True) + 1e-8)
        return delta_z
