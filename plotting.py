# colored scatterplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path


def plot_square_dw_scatter(
    csv_path,
    outpath,
    pred_col="y_pred_top32",
    true_col="y_true",
    pad=0.05,
):
    df = pd.read_csv(csv_path)

    x = df[pred_col].to_numpy()
    y = df[true_col].to_numpy()

    # correlation for annotation
    rho, _ = spearmanr(y, x)

    # use same limits for both axes
    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max())
    span = vmax - vmin
    lo = vmin - pad * span
    hi = vmax + pad * span

    fig, ax = plt.subplots(figsize=(7, 7))

    sc = ax.scatter(
        x,
        y,
        c=y,                   # color by true DW score
        cmap="bwr",            # blue-white-red
        vmin=-1,
        vmax=1,
        s=50,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.8,
    )

    # diagonal line y = x
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="gray")

    # square axes with same scale
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("Actual", fontsize=14)

    # paper-style rho annotation
    ax.text(
        0.97,
        0.02,
        rf"$\rho^{{CV}}_{{K=32}} = {rho:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=18,
    )

    #cbar = plt.colorbar(sc, ax=ax)
    #cbar.set_label("True DW-NOMINATE", fontsize=12)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    outdir = Path("mistral_cfs_outputs")
    plot_square_dw_scatter(
        csv_path=outdir / "topk_ensemble_predictions.csv",
        outpath=outdir / "figure3_style_top32_scatter.png",
        pred_col="y_pred_top32",
        true_col="y_true",
    )
    print("Saved to:", outdir / "figure3_style_top32_scatter.png")