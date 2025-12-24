import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

class RicciFlow:
    """
    Simple Ricci flow smoothing for metric tensors
    Based on discrete Ollivier-Ricci curvature approximation
    """
    def __init__(self, k_neighbors=15, smoothing_factor=0.3, iterations=3):
        self.k = k_neighbors
        self.alpha = smoothing_factor
        self.iterations = iterations
    
    def compute_curvature_weights(self, Z):
        """
        Compute curvature-based weights for smoothing
        Z: latent codes (n_samples, latent_dim)
        Returns: weights matrix (n_samples, n_samples)
        """
        n_samples = Z.shape[0]
        
        if n_samples < self.k + 1:
            # Not enough samples, return uniform weights
            return torch.ones(n_samples, n_samples) / n_samples
        
        # Convert to numpy for sklearn
        Z_np = Z.cpu().numpy()
        
        # Build k-NN graph
        knn = NearestNeighbors(n_neighbors=min(self.k + 1, n_samples))
        knn.fit(Z_np)
        distances, indices = knn.kneighbors(Z_np)
        
        # Create weight matrix based on distances
        weights = torch.zeros(n_samples, n_samples)
        
        for i in range(n_samples):
            # Get neighbors (excluding self)
            neighbors = indices[i, 1:]
            neighbor_dists = distances[i, 1:]
            
            # Convert distances to similarities (inverse)
            if len(neighbor_dists) > 0:
                similarities = 1.0 / (1.0 + neighbor_dists)
                similarities = similarities / similarities.sum()
                
                for j, neighbor_idx in enumerate(neighbors):
                    weights[i, neighbor_idx] = similarities[j]
        
        return weights
    
    def smooth_metrics(self, Z, G_diags):
        """
        Apply Ricci flow smoothing to diagonal metrics
        Z: latent codes (n_samples, latent_dim)
        G_diags: diagonal of metric tensors (n_samples, data_dim)
        Returns: smoothed metrics
        """
        if self.iterations == 0:
            return G_diags
        
        print(f"Applying Ricci flow smoothing (k={self.k}, iterations={self.iterations})")
        
        # Compute curvature weights
        weights = self.compute_curvature_weights(Z)
        
        G_smoothed = G_diags.clone()
        
        # Apply smoothing iterations
        for iter in range(self.iterations):
            # Weighted average with neighbors
            G_weighted = torch.matmul(weights, G_smoothed)
            
            # Blend with original
            G_smoothed = (1 - self.alpha) * G_smoothed + self.alpha * G_weighted
            
            # Ensure positivity
            G_smoothed = torch.clamp(G_smoothed, min=1e-6)
            
            if (iter + 1) % 2 == 0:
                print(f"  Ricci flow iteration {iter + 1}/{self.iterations}")
        
        return G_smoothed