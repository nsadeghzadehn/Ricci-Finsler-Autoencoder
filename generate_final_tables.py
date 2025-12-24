# generate_final_tables.py (final working version)
import json
from pathlib import Path
import pandas as pd

# =========================
# Paths (run from project root)
# =========================
outputs_dir = Path('outputs')
tables_dir = Path('tables')
tables_dir.mkdir(exist_ok=True)

json_path = outputs_dir / 'multi_seed_results.json'
csv_path = outputs_dir / 'ricci_sweep_results.csv'

if not json_path.exists():
    raise FileNotFoundError(f"Missing file: {json_path.resolve()}")

if not csv_path.exists():
    raise FileNotFoundError(f"Missing file: {csv_path.resolve()}")

# =========================
# Load JSON results
# =========================
def load_results():
    with open(json_path, 'r') as f:
        return json.load(f)

# =========================
# Main dataset tables
# =========================
def create_latex_table(dataset_name, dataset_stats, include_dir_sim=True):
    lines = []

    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Experimental results on \texttt{{{dataset_name}}} dataset (mean $\pm$ std)}}"
    )
    lines.append(rf"\label{{tab:results_{dataset_name}}}")

    if include_dir_sim:
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Dataset} & \textbf{MSE} & \textbf{Trust} & \textbf{DirSim} & \textbf{Loss} \\")
    else:
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Dataset} & \textbf{MSE} & \textbf{Trust} & \textbf{Loss} \\")

    lines.append(r"\midrule")

    mse = f"{dataset_stats.get('MSE_mean', 0):.4f} $\\pm$ {dataset_stats.get('MSE_std', 0):.4f}"
    trust = f"{dataset_stats.get('Trust_mean', dataset_stats.get('trust', 0)):.3f}"
    loss = f"{dataset_stats.get('Loss_mean', dataset_stats.get('loss', 0)):.4f}"

    if include_dir_sim:
        dir_sim = f"{dataset_stats.get('DirSim_mean', dataset_stats.get('dir_sim', 0)):.3f}"
        line = f"{dataset_name.capitalize()} & {mse} & {trust} & {dir_sim} & {loss} \\\\"
    else:
        line = f"{dataset_name.capitalize()} & {mse} & {trust} & {loss} \\\\"

    lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

# =========================
# Ricci sweep table
# =========================
def create_ricci_sweep_table():
    df = pd.read_csv(csv_path)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ricci parameter sweep results (synthetic dataset)}")
    lines.append(r"\label{tab:ricci_sweep}")
    lines.append(r"\begin{tabular}{ccccc}")
    lines.append(r"\toprule")
    lines.append(r"$k$ & Iter & Loss & MSE & Trust \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        lines.append(
            f"{int(row['ricci_k'])} & "
            f"{int(row['ricci_iter'])} & "
            f"{row['loss']:.4f} & "
            f"{row['mse']:.4f} & "
            f"{row['trust']:.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

# =========================
# Main
# =========================
def main():
    print("Generating LaTeX tables...")

    results = load_results()

    for dataset_name, dataset_stats in results.items():
        tex = create_latex_table(dataset_name, dataset_stats)
        out_path = tables_dir / f"table_{dataset_name}_final.tex"
        out_path.write_text(tex, encoding='utf-8')
        print(f"  ✔ {out_path.name}")

    ricci_tex = create_ricci_sweep_table()
    ricci_path = tables_dir / "table_ricci_sweep.tex"
    ricci_path.write_text(ricci_tex, encoding='utf-8')
    print(f"  ✔ {ricci_path.name}")

    print("\n✅ All tables generated successfully")
    print(f"📁 Tables directory: {tables_dir.resolve()}")

if __name__ == "__main__":
    main()
