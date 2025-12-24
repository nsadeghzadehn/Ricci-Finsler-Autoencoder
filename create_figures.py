# create_figures.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =========================
# Global settings
# =========================
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# =========================
# Paths (run from project root)
# =========================
outputs_dir = Path('outputs')
figures_dir = Path('figures')

figures_dir.mkdir(exist_ok=True)

csv_path = outputs_dir / 'ricci_sweep_results.csv'

if not csv_path.exists():
    raise FileNotFoundError(f"Ricci sweep file not found: {csv_path.resolve()}")

# =========================
# Load data
# =========================
df = pd.read_csv(csv_path)

# =========================
# Figure 1: MSE heatmap
# =========================
plt.figure(figsize=(8, 6))
heatmap_data = df.pivot(index='ricci_k', columns='ricci_iter', values='mse')
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.4f',
    cmap='YlOrRd',
    cbar_kws={'label': 'MSE'}
)
plt.title('MSE Heatmap for Ricci Parameter Sweep', fontsize=14)
plt.xlabel('Ricci Iterations')
plt.ylabel('Number of Neighbors (k)')
plt.tight_layout()
plt.savefig(figures_dir / 'ricci_mse_heatmap.png')
plt.close()

# =========================
# Figure 2: Loss curves
# =========================
plt.figure(figsize=(10, 6))
for k in sorted(df['ricci_k'].unique()):
    subset = df[df['ricci_k'] == k].sort_values('ricci_iter')
    plt.plot(
        subset['ricci_iter'],
        subset['loss'],
        marker='o',
        linewidth=2,
        label=f'k={k}'
    )

plt.xlabel('Ricci Iterations', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Across Ricci Iterations for Different k Values', fontsize=14)
plt.legend(title='Number of Neighbors')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'ricci_loss_curve.png')
plt.close()

# =========================
# Figure 3: Trustworthiness curves
# =========================
plt.figure(figsize=(10, 6))
for k in sorted(df['ricci_k'].unique()):
    subset = df[df['ricci_k'] == k].sort_values('ricci_iter')
    plt.plot(
        subset['ricci_iter'],
        subset['trust'],
        marker='s',
        linewidth=2,
        label=f'k={k}'
    )

plt.xlabel('Ricci Iterations', fontsize=12)
plt.ylabel('Trustworthiness', fontsize=12)
plt.title('Trustworthiness Across Ricci Iterations for Different k Values', fontsize=14)
plt.legend(title='Number of Neighbors')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'ricci_trust_curve.png')
plt.close()

print("✅ All figures generated successfully.")
print(f"📁 Figures saved to: {figures_dir.resolve()}")
