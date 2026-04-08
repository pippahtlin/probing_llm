#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCORES = ["dw", "cfs", "bt"]
PRETTY = {"dw": "DW", "cfs": "CFS", "bt": "BT"}

BASE_DIR = Path("/home/pippalin2/probing_llm/intervention_outputs_v2")

TARGET_RANGES = {
    "dw": (-1.0, 1.0),
    "cfs": (-2.0, 2.0),
    "bt": (-1.0, 1.0),
}


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def sem(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def load_data(input_dir: Path) -> pd.DataFrame:
    agg_path = input_dir / "aggregate_scores.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing {agg_path}")

    df = pd.read_csv(agg_path)

    required = {
        "source_score", "head_mode", "topic", "alpha",
        "dw_pred_mean", "cfs_pred_mean", "bt_pred_mean"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"aggregate_scores.csv is missing columns: {sorted(missing)}")

    df["source_score"] = df["source_score"].str.lower()
    df["head_mode"] = df["head_mode"].str.lower()
    df["topic"] = df["topic"].astype(str)

    return df


def rescale_to_semantic_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linearly rescale each score family to its semantic range.
    No clipping.
    """
    df = df.copy()

    for score in SCORES:
        raw_col = f"{score}_pred_mean"
        scaled_col = f"{score}_scaled"

        lo_tgt, hi_tgt = TARGET_RANGES[score]
        lo_raw = df[raw_col].min()
        hi_raw = df[raw_col].max()

        if not np.isfinite(lo_raw) or not np.isfinite(hi_raw) or hi_raw == lo_raw:
            df[scaled_col] = 0.0
        else:
            df[scaled_col] = lo_tgt + (df[raw_col] - lo_raw) * (hi_tgt - lo_tgt) / (hi_raw - lo_raw)

    return df


def add_delta_from_alpha0(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within each (source_score, head_mode, topic) curve, subtract alpha=0 baseline.
    """
    df = df.copy()
    group_cols = ["source_score", "head_mode", "topic"]
    out_parts = []

    for _, g in df.groupby(group_cols, sort=False):
        g = g.copy().sort_values("alpha")

        base_rows = g[g["alpha"] == 0]
        if len(base_rows) == 0:
            idx0 = np.argmin(np.abs(g["alpha"].to_numpy(dtype=float)))
            base_row = g.iloc[idx0]
        else:
            base_row = base_rows.iloc[0]

        for score in SCORES:
            scaled_col = f"{score}_scaled"
            delta_col = f"{score}_delta"
            g[delta_col] = g[scaled_col] - float(base_row[scaled_col])

        out_parts.append(g)

    return pd.concat(out_parts, axis=0, ignore_index=True)


def build_effect_tables(df: pd.DataFrame, outdir: Path):
    rows = []
    topic_rows = []

    topk = df[df["head_mode"] == "topk"].copy()

    for source in SCORES:
        sub_source = topk[topk["source_score"] == source]
        for target in SCORES:
            score_col = f"{target}_delta"
            alpha_curve = (
                sub_source.groupby("alpha", as_index=False)[score_col]
                .mean()
                .sort_values("alpha")
            )
            slope = fit_slope(alpha_curve["alpha"].to_numpy(), alpha_curve[score_col].to_numpy())
            effect = (
                alpha_curve.iloc[-1][score_col] - alpha_curve.iloc[0][score_col]
                if len(alpha_curve) >= 2 else np.nan
            )
            rows.append({
                "source_score": source,
                "target_score": target,
                "slope": slope,
                "effect_max_minus_min": effect,
            })

    for source in SCORES:
        score_col = f"{source}_delta"
        sub_source = topk[topk["source_score"] == source]
        for topic, sub_topic in sub_source.groupby("topic"):
            curve = sub_topic.sort_values("alpha")
            topic_rows.append({
                "source_score": source,
                "topic": topic,
                "matched_slope": fit_slope(
                    curve["alpha"].to_numpy(), curve[score_col].to_numpy()
                ),
                "matched_effect_max_minus_min": (
                    curve.iloc[-1][score_col] - curve.iloc[0][score_col]
                    if len(curve) >= 2 else np.nan
                ),
            })

    effect_df = pd.DataFrame(rows)
    topic_effect_df = pd.DataFrame(topic_rows)

    effect_df.to_csv(outdir / "cross_effect_summary_semantic.csv", index=False)
    topic_effect_df.to_csv(outdir / "topic_effect_summary_semantic.csv", index=False)

    return effect_df, topic_effect_df


def plot_cross_effect_matrix(effect_df: pd.DataFrame, outdir: Path):
    mat = np.zeros((3, 3), dtype=float)

    for i, source in enumerate(SCORES):
        for j, target in enumerate(SCORES):
            val = effect_df.loc[
                (effect_df["source_score"] == source) & (effect_df["target_score"] == target),
                "slope"
            ].iloc[0]
            mat[i, j] = val

    vmax = np.nanmax(np.abs(mat))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="equal")

    ax.set_xticks(range(3), [PRETTY[s] for s in SCORES])
    ax.set_yticks(range(3), [PRETTY[s] for s in SCORES])
    ax.set_xlabel("Probe used for scoring")
    ax.set_ylabel("Intervention source")
    ax.set_title("3×3 cross-effect matrix (semantic Δ from α=0)")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=11)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Slope of Δscore vs alpha")

    fig.tight_layout()
    fig.savefig(outdir / "cross_effect_matrix_semantic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure4a_raw(df: pd.DataFrame, outdir: Path):
    topk = df[df["head_mode"] == "topk"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)

    global_ymin = min(v[0] for v in TARGET_RANGES.values())
    global_ymax = max(v[1] for v in TARGET_RANGES.values())

    for ax, source in zip(axes, SCORES):
        sub = topk[topk["source_score"] == source]
        for target in SCORES:
            score_col = f"{target}_scaled"
            mean_curve = sub.groupby("alpha", as_index=False)[score_col].mean().sort_values("alpha")
            err = (
                sub.groupby("alpha")[score_col]
                .apply(lambda s: sem(s.to_numpy()))
                .reindex(mean_curve["alpha"])
                .to_numpy()
            )
            ax.plot(mean_curve["alpha"], mean_curve[score_col], marker="o", label=PRETTY[target])
            ax.fill_between(
                mean_curve["alpha"].to_numpy(dtype=float),
                mean_curve[score_col].to_numpy(dtype=float) - err,
                mean_curve[score_col].to_numpy(dtype=float) + err,
                alpha=0.18,
            )

        ax.set_title(f"{PRETTY[source]} intervention")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Rescaled political score")
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_ylim(global_ymin - 0.1, global_ymax + 0.1)

    axes[0].legend(frameon=False)
    fig.suptitle("Figure 4a style: raw rescaled political scores", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "figure4a_three_panel_semantic_raw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure4a_delta(df: pd.DataFrame, outdir: Path):
    topk = df[df["head_mode"] == "topk"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    for ax, source in zip(axes, SCORES):
        sub = topk[topk["source_score"] == source]
        for target in SCORES:
            score_col = f"{target}_delta"
            mean_curve = sub.groupby("alpha", as_index=False)[score_col].mean().sort_values("alpha")
            err = (
                sub.groupby("alpha")[score_col]
                .apply(lambda s: sem(s.to_numpy()))
                .reindex(mean_curve["alpha"])
                .to_numpy()
            )
            ax.plot(mean_curve["alpha"], mean_curve[score_col], marker="o", label=PRETTY[target])
            ax.fill_between(
                mean_curve["alpha"].to_numpy(dtype=float),
                mean_curve[score_col].to_numpy(dtype=float) - err,
                mean_curve[score_col].to_numpy(dtype=float) + err,
                alpha=0.18,
            )

        ax.set_title(f"{PRETTY[source]} intervention")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Δ score from α=0")
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.axhline(0, linestyle="--", linewidth=1, alpha=0.7)

    axes[0].legend(frameon=False)
    fig.suptitle("Figure 4a style: semantic score shift vs intervention strength", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "figure4a_three_panel_semantic_delta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure4b_topic_effect(topic_effect_df: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, source in zip(axes, SCORES):
        sub = topic_effect_df[topic_effect_df["source_score"] == source].copy()
        sub = sub.sort_values("matched_slope", ascending=False)
        ax.bar(range(len(sub)), sub["matched_slope"].to_numpy())
        ax.set_xticks(range(len(sub)), sub["topic"].tolist(), rotation=35, ha="right")
        ax.set_title(f"{PRETTY[source]} intervention")
        ax.set_xlabel("Topic")
        ax.axhline(0, linestyle="--", linewidth=1)

    axes[0].set_ylabel("Slope of matched Δscore vs alpha")
    fig.suptitle("Figure 4b style: topic-specific matched effects", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "figure4b_topic_effect_semantic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_topk_vs_random(df: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

    for ax, source in zip(axes, SCORES):
        score_col = f"{source}_delta"
        for mode in ["topk", "random"]:
            sub = df[(df["source_score"] == source) & (df["head_mode"] == mode)].copy()
            curve = sub.groupby("alpha", as_index=False)[score_col].mean().sort_values("alpha")
            err = (
                sub.groupby("alpha")[score_col]
                .apply(lambda s: sem(s.to_numpy()))
                .reindex(curve["alpha"])
                .to_numpy()
            )
            label = "Topk" if mode == "topk" else "Random"
            ax.plot(curve["alpha"], curve[score_col], marker="o", label=label)
            ax.fill_between(
                curve["alpha"].to_numpy(dtype=float),
                curve[score_col].to_numpy(dtype=float) - err,
                curve[score_col].to_numpy(dtype=float) + err,
                alpha=0.18,
            )

        ax.set_title(f"{PRETTY[source]} intervention")
        ax.set_xlabel("alpha")
        ax.set_ylabel(f"{PRETTY[source]} Δscore")
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.axhline(0, linestyle="--", linewidth=1, alpha=0.7)

    axes[0].legend(frameon=False)
    fig.suptitle("Topk vs Random sanity check (matched semantic Δscore)", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "topk_vs_random_semantic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    input_dir = BASE_DIR
    outdir = BASE_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_dir)
    df = rescale_to_semantic_ranges(df)
    df = add_delta_from_alpha0(df)

    effect_df, topic_effect_df = build_effect_tables(df, outdir)

    plot_cross_effect_matrix(effect_df, outdir)
    plot_figure4a_raw(df, outdir)
    plot_figure4a_delta(df, outdir)
    plot_figure4b_topic_effect(topic_effect_df, outdir)
    plot_topk_vs_random(df, outdir)

    print("Saved:")
    for name in [
        "cross_effect_summary_semantic.csv",
        "topic_effect_summary_semantic.csv",
        "cross_effect_matrix_semantic.png",
        "figure4a_three_panel_semantic_raw.png",
        "figure4a_three_panel_semantic_delta.png",
        "figure4b_topic_effect_semantic.png",
        "topk_vs_random_semantic.png",
    ]:
        print(outdir / name)


if __name__ == "__main__":
    main()