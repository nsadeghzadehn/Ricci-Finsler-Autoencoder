import torch

def safe_quad(h, G_diag=None, eps=1e-12):
    """
    If G_diag is provided (B, D) -> use elementwise: q = sum(G_diag * h^2)
    else fallback to Euclidean norm squared.
    Returns sqrt(q) (alpha term).
    """
    if G_diag is not None:
        q = torch.sum(G_diag * (h ** 2), dim=1)  # (B,)
    else:
        q = torch.sum(h * h, dim=1)
    q = torch.clamp(q, min=eps)
    return torch.sqrt(q)

def project_beta(beta, Ginv_diag=None, cap=0.9):
    """
    If using diagonal Ginv: enforce ||beta||_{Ginv} < cap
    Ginv_diag: (B, D) diag of G^{-1}
    Returns projected beta (B,D)
    """
    if Ginv_diag is None:
        # Euclidean norm cap
        n = torch.norm(beta, dim=1)  # (B,)
        scale = torch.clamp(cap / (n + 1e-12), max=1.0)
        return beta * scale.unsqueeze(1)
    else:
        n2 = torch.sum(Ginv_diag * (beta ** 2), dim=1)
        n = torch.sqrt(torch.clamp(n2, min=1e-12))
        scale = torch.clamp(cap / (n + 1e-12), max=1.0)
        return beta * scale.unsqueeze(1)

def finsler_per_sample(h, beta, G_diag=None, Ginv_diag=None, beta_mode='limited', cap=0.9):
    """
    h: (B,D) residual
    beta: (B,D) raw BetaNet output
    G_diag: (B,D) diagonal of G (optional)
    Ginv_diag: (B,D) diagonal of G^{-1} (optional)
    beta_mode: 'zero', 'free', 'limited'
    """
    alpha_term = safe_quad(h, G_diag)  # (B,)
    if beta_mode == 'zero':
        beta_proj = torch.zeros_like(h)
    elif beta_mode == 'free':
        beta_proj = beta
    else:  # limited
        beta_proj = project_beta(beta, Ginv_diag, cap=cap)
    beta_term = torch.sum(beta_proj * h, dim=1)  # (B,)
    Fvals = alpha_term + beta_term
    Fvals = torch.clamp(Fvals, min=1e-9)
    return Fvals, alpha_term, beta_term, beta_proj

def compute_finsler_loss(h, beta, g_diag=None, ginv_diag=None, 
                        beta_mode='limited', cap=0.9, eps=1e-12):
    """
    تابع loss کامل فینسلر
    h: خطای بازسازی (B, D)
    beta: خروجی BetaNet (B, D)
    """
    # محاسبه Finsler value
    F_vals, alpha_term, beta_term, beta_proj = finsler_per_sample(
        h, beta, g_diag, ginv_diag, beta_mode, cap
    )
    
    # loss نهایی
    loss = F_vals.mean()
    
    return {
        'total_loss': loss,
        'finsler_values': F_vals,
        'alpha_term': alpha_term.mean(),
        'beta_term': beta_term.mean(),
        'beta_proj': beta_proj
    }