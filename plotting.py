# plot_ensemble_robustness.py
# --------------------------------
# Reads an ensemble robustness CSV and generates two plots:
#   1) PGD attack robustness (Top-1 vs iterations)
#   2) FGSM attack robustness (Top-1 vs epsilon)

import re
import pandas as pd
import matplotlib.pyplot as plt


def plot_robustness(csv_path: str, save_dir: str = None):
    """
    Loads an ensemble robustness CSV and plots:
      - PGD robustness: Top-1 accuracy (%) vs PGD iterations
      - FGSM robustness: Top-1 accuracy (%) vs FGSM epsilon
    Args:
        csv_path: Path to the CSV file
        save_dir: Directory to save figures (if provided)
    """
    # Load data
    df = pd.read_csv(csv_path)
    models = df['Model'].tolist()

    # Identify PGD columns: e.g. 'PGD_iters=7_eps0.031'
    pgd_cols = [c for c in df.columns if c.startswith('PGD_') and 'iters' in c]
    # Identify FGSM columns: e.g. 'FGSM_eps=0.03'
    fgsm_cols = [c for c in df.columns if c.startswith('FGSM_') and 'eps' in c]

    if not pgd_cols and not fgsm_cols:
        raise ValueError("No PGD or FGSM columns found in CSV.")

    # Parse PGD iters and eps
    pgd_info = []
    pgd_eps = None
    for col in pgd_cols:
        m_iter = re.search(r"iters=(\d+)", col)
        m_eps  = re.search(r"eps([0-9\.]+)", col)
        if m_iter:
            iters = int(m_iter.group(1))
            pgd_info.append((col, iters))
            if pgd_eps is None and m_eps:
                pgd_eps = float(m_eps.group(1))
    pgd_info.sort(key=lambda x: x[1])
    pgd_cols_sorted = [c for c, _ in pgd_info]
    pgd_iters = [it for _, it in pgd_info]

    # Parse FGSM epsilons
    fgsm_info = []
    for col in fgsm_cols:
        m_eps = re.search(r"eps=([0-9\.]+)", col)
        if m_eps:
            eps = float(m_eps.group(1))
            fgsm_info.append((col, eps))
    fgsm_info.sort(key=lambda x: x[1])
    fgsm_cols_sorted = [c for c, _ in fgsm_info]
    fgsm_eps = [eps for _, eps in fgsm_info]

    # Determine if values are fractions
    sample_vals = []
    if pgd_cols_sorted:
        sample_vals = df[pgd_cols_sorted].iloc[0].values
    elif fgsm_cols_sorted:
        sample_vals = df[fgsm_cols_sorted].iloc[0].values
    scale = 100.0 if all(v <= 1.0 for v in sample_vals) else 1.0

    # Create subplots
    n_plots = (1 if not fgsm_cols_sorted else 2) if pgd_cols_sorted else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5), squeeze=False)

    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    # PGD plot (left)
    if pgd_cols_sorted:
        ax = axes[0,0]
        for idx, model in enumerate(models):
            y = df.loc[idx, pgd_cols_sorted] * scale
            ax.plot(pgd_iters, y,
                    marker=markers[idx % len(markers)], linewidth=2, label=model)
        ax.set_xlabel('PGD Attack Iterations', fontsize=12)
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        title = f"PGD Attack (ε={pgd_eps}) Robustness" if pgd_eps else "PGD Attack Robustness"
        ax.set_title(title, fontsize=14)
        ax.set_xticks(pgd_iters)
        ax.set_ylim(0,100)
        ax.legend(title='Defense', loc='upper right', fontsize=10)

    # FGSM plot (right)
    if fgsm_cols_sorted and n_plots == 2:
        ax = axes[0,1]
        for idx, model in enumerate(models):
            y = df.loc[idx, fgsm_cols_sorted] * scale
            ax.plot(fgsm_eps, y,
                    marker=markers[idx % len(markers)], linewidth=2, label=model)
        ax.set_xlabel('FGSM Epsilon', fontsize=12)
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax.set_title('FGSM Attack Robustness', fontsize=14)
        ax.set_xticks(fgsm_eps)
        ax.set_ylim(0,100)
        ax.legend(title='Defense', loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save figures if requested
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        pgd_fname = os.path.join(save_dir, 'ensemble_pgd_robustness.png')
        plt.sca(axes[0,0])
        plt.savefig(pgd_fname, dpi=300, bbox_inches='tight')
        if fgsm_cols_sorted:
            fgsm_fname = os.path.join(save_dir, 'ensemble_fgsm_robustness.png')
            plt.sca(axes[0,1])
            plt.savefig(fgsm_fname, dpi=300, bbox_inches='tight')
        print(f"Saved plots to {save_dir}")

    plt.show()


if __name__ == '__main__':
    # Example usage
    csv_file = '/home/work/AIprogramming/Ai_proj/robustness_results/CIFAR10_all_ensemble_robustness_summary_compat.csv'
    out_dir = '/home/work/AIprogramming/Ai_proj/graphs'
    plot_robustness(csv_file, save_dir=out_dir)
