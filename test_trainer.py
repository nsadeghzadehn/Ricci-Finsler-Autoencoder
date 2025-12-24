# test_trainer.py - VERSION FOR NEW ENVIRONMENT

import torch
import itertools
import pandas as pd
import sys
import os

# =============================
# FIX 1: Add project root to Python path for proper imports
# =============================
sys.path.append('..')  # This allows importing from core/

# Now import the trainer
try:
    from core.trainer_complete import FinslerTrainer
    print("✅ Successfully imported FinslerTrainer from core module")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the scripts/ directory and core/ folder exists")
    sys.exit(1)

# =============================
# STEP 0: Dataset preparation
# =============================
num_samples = 1000   # تعداد نمونه‌ها
data_dim = 100       # بعد ویژگی‌ها

# دیتاست مصنوعی ژورنال پسند (مثلاً Gaussian blobs)
torch.manual_seed(42)
X = torch.randn(num_samples, data_dim) * 0.5 + 0.0

# FIX 2: Save dataset shape in the project root, not scripts/
dataset_shape_path = os.path.join('..', 'dataset_shape.txt')
with open(dataset_shape_path, "w") as f:
    f.write(f"Dataset shape: {X.shape}\n")

print(f"Dataset prepared. Shape: {X.shape}")
print(f"Dataset shape logged in {dataset_shape_path}")

# =============================
# STEP 1: Config
# =============================
config = {
    'data_dim': data_dim,
    'latent_dim': 8,
    'batch_size': 32,
    'epochs': 5,
    'lr': 0.0005,
    'diagonal_only': True,
    'beta_mode': 'limited',
    'beta_cap': 0.5,
    'beta_schedule': [0.00001, 0.0001, 0.001, 0.01],
    'finsler_schedule': [0.00001, 0.0001, 0.001, 0.005],
    # فعال کردن Ricci flow
    'use_ricci': True,
    'ricci_k': 10,
    'ricci_alpha': 0.15,
    'ricci_iter': 2,
}

# پارامترهای sweep
ricci_ks = [5, 10, 20]
ricci_iters = [1, 2, 4]

# =============================
# STEP 2: Ricci sweep
# =============================
param_combinations = list(itertools.product(ricci_ks, ricci_iters))
print(f"Total Ricci sweep combinations: {len(param_combinations)}")

results_list = []

for idx, (k_val, iter_val) in enumerate(param_combinations, 1):
    print(f"\n[{idx}/{len(param_combinations)}] Running Ricci sweep: k={k_val}, iterations={iter_val}")
    
    # تغییر config برای هر run
    config['ricci_k'] = k_val
    config['ricci_iter'] = iter_val
    
    # ایجاد trainer جدید
    trainer = FinslerTrainer(config)
    
    # اجرای training
    result = trainer.run_all(X)
    
    # ذخیره نتایج اصلی
    results_list.append({
        'ricci_k': k_val,
        'ricci_iter': iter_val,
        'loss': result['final']['loss'],
        'mse': result['final']['mse'],
        'trust': result['final']['trust']
    })

# =============================
# STEP 3: Save results
# =============================
# FIX 3: Save CSV in the outputs/ folder
outputs_dir = os.path.join('..', 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

csv_path = os.path.join(outputs_dir, 'ricci_sweep_results.csv')
df_results = pd.DataFrame(results_list)
df_results.to_csv(csv_path, index=False)

print("\n" + "="*60)
print("ALL RICCI SWEEP RUNS COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Results saved to: {csv_path}")
print(f"Total runs: {len(results_list)}")
print(f"Parameter space: k ∈ {ricci_ks}, iterations ∈ {ricci_iters}")
print("="*60)

# Display a summary of results
print("\nSUMMARY OF RESULTS:")
print(df_results.to_string(index=False))