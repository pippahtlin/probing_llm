#!/usr/bin/env python3
"""
Pipeline for:
1) reading a CFS CSV,
2) recording prompts for the first 5 senators,
3) extracting per-layer and per-head activations from Mistral-7B-Instruct,
4) training ridge linear probes for recipient CFscore,
5) computing Spearman correlation for every layer/head,
6) saving results and plotting a heatmap,
7) saving paper-style Top-K ensemble results.

Usage example:
python mistral_cfs_pipeline.py \
    --cfs-csv /home/pippalin2/probing_llm/data/116th_Senate_CFS.csv \
    --outdir ./mistral_cfs_outputs

Notes:
- This script follows the activation-capture pattern in the user's mistral_test.py:
  it hooks the INPUT to each layer's self_attn.o_proj and averages over tokens.
- It uses 2-fold cross-validation and ridge probes, following the linear-probe setup
  described in the attached political representation paper.
- Following the Bradley-Terry paper, the default target is recipient.cfscore.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEFAULT_CACHE_DIR = "/net/scratch/pippalin2/hf_cache"

NUM_LAYERS = 32
NUM_HEADS = 32
HIDDEN_SIZE = 4096
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS


@dataclass
class Record:
    senator: str
    prompt: str
    cfscore: float
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    layer_hidden_avg: torch.Tensor   # [32, 4096]
    head_avg: torch.Tensor           # [32, 32, 128]


def normalize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def infer_name_column(df: pd.DataFrame) -> str:
    candidates = [
        "senator",
        "name",
        "full_name",
        "bioname",
        "member_name",
        "person_name",
        "legislator",
        "lawmaker",
        "recipient.name",
        "recipient",
        "candidate",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        raise ValueError("Could not infer senator-name column in CFS CSV.")
    return max(obj_cols, key=lambda c: df[c].nunique())


def infer_cfscore_column(df: pd.DataFrame) -> str:
    # Bradley-Terry paper uses recipient CFscore.
    exact_candidates = [
        "recipient.cfscore",
        "recipient_cfscore",
        "recipient cfscore",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in exact_candidates:
        if c in lower_map:
            return lower_map[c]

    fallback_candidates = [
        c for c in df.columns
        if "cfscore" in c.lower() and "recipient" in c.lower()
    ]
    if fallback_candidates:
        return fallback_candidates[0]

    raise ValueError(
        "Could not find recipient CFscore column. "
        "Expected something like 'recipient.cfscore'."
    )


def load_cfs_data(csv_path: Path, first_n: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    name_col = infer_name_column(df)
    cfscore_col = infer_cfscore_column(df)

    out = df[[name_col, cfscore_col]].copy()
    out.columns = ["senator", "cfscore"]
    out["senator"] = out["senator"].map(normalize_name)
    out["cfscore"] = pd.to_numeric(out["cfscore"], errors="coerce")
    out = out.dropna(subset=["senator", "cfscore"]).drop_duplicates(subset=["senator"])

    if first_n is not None:
        out = out.head(first_n).copy()

    return out.reset_index(drop=True)


def build_prompt(senator: str) -> str:
    return (
        f"[INST] Write a political statement that sounds like U.S. Senator {senator}. [/INST]"
    )


def load_model(cache_dir: str = DEFAULT_CACHE_DIR):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model.eval()
    return tokenizer, model


def extract_records(
    examples_df: pd.DataFrame,
    tokenizer,
    model,
    outdir: Path,
) -> List[Record]:
    captured_head_outputs: Dict[int, torch.Tensor] = {}

    def make_o_proj_pre_hook(layer_idx: int):
        def hook(module, inputs):
            x = inputs[0].detach().cpu()  # [B, T, H]
            captured_head_outputs[layer_idx] = x
        return hook

    hooks = []
    for layer_idx in range(NUM_LAYERS):
        hook = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
            make_o_proj_pre_hook(layer_idx)
        )
        hooks.append(hook)

    records: List[Record] = []
    model_device = next(model.parameters()).device

    try:
        for _, row in examples_df.iterrows():
            senator = row["senator"]
            cfscore = float(row["cfscore"])
            prompt = build_prompt(senator)

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            captured_head_outputs.clear()

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            layer_hidden_avg = []
            for layer_idx in range(1, len(outputs.hidden_states)):  # skip embedding output
                h = outputs.hidden_states[layer_idx].detach().cpu()   # [B, T, H]
                h_avg = h.mean(dim=1).squeeze(0)                      # [H]
                layer_hidden_avg.append(h_avg)
            layer_hidden_avg = torch.stack(layer_hidden_avg, dim=0)   # [32, 4096]

            head_avg_all_layers = []
            for layer_idx in range(NUM_LAYERS):
                x = captured_head_outputs[layer_idx]                  # [B, T, 4096]
                bsz, tsz, hsz = x.shape
                if hsz != HIDDEN_SIZE:
                    raise ValueError(f"Unexpected hidden size at layer {layer_idx}: {hsz}")
                x = x.view(bsz, tsz, NUM_HEADS, HEAD_DIM)            # [B, T, 32, 128]
                x_avg = x.mean(dim=1).squeeze(0)                     # [32, 128]
                head_avg_all_layers.append(x_avg)
            head_avg_all_layers = torch.stack(head_avg_all_layers, dim=0)  # [32, 32, 128]

            records.append(
                Record(
                    senator=senator,
                    prompt=prompt,
                    cfscore=cfscore,
                    input_ids=inputs["input_ids"].detach().cpu(),
                    attention_mask=inputs["attention_mask"].detach().cpu(),
                    layer_hidden_avg=layer_hidden_avg,
                    head_avg=head_avg_all_layers,
                )
            )
    finally:
        for h in hooks:
            h.remove()

    torch.save(records, outdir / "all_records.pt")
    return records


def save_first_five_prompts(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    first5 = df.head(5).copy()
    first5["prompt"] = first5["senator"].map(build_prompt)
    first5.to_csv(outdir / "first5_senator_prompts.csv", index=False)
    return first5


def flatten_features(records: Sequence[Record]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(records)
    X = np.zeros((n, NUM_LAYERS, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    for i, rec in enumerate(records):
        X[i] = rec.head_avg.numpy()
        y[i] = rec.cfscore
    return X, y


def cv_spearman_for_head(
    X_head: np.ndarray,
    y: np.ndarray,
    n_splits: int = 2,
    alpha: float = 1.0,
    seed: int = 42,
) -> float:
    if len(y) < n_splits:
        return np.nan

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.full_like(y, fill_value=np.nan, dtype=np.float32)

    for train_idx, test_idx in kf.split(X_head):
        X_train, X_test = X_head[train_idx], X_head[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        preds[test_idx] = model.predict(X_test)

    mask = ~np.isnan(preds)
    if mask.sum() < 3:
        return np.nan
    rho, _ = spearmanr(y[mask], preds[mask])
    return float(rho)


def compute_headwise_correlations(
    records: Sequence[Record],
    outdir: Path,
    alpha: float = 1.0,
    n_splits: int = 2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    X, y = flatten_features(records)
    corr = np.zeros((NUM_LAYERS, NUM_HEADS), dtype=np.float32)

    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            X_head = X[:, layer, head, :]   # [N, 128]
            corr[layer, head] = cv_spearman_for_head(
                X_head=X_head,
                y=y,
                n_splits=n_splits,
                alpha=alpha,
                seed=seed,
            )

    rows = []
    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            rows.append(
                {
                    "layer": layer + 1,
                    "head": head + 1,
                    "spearman_cv": float(corr[layer, head]),
                }
            )

    corr_df = pd.DataFrame(rows).sort_values("spearman_cv", ascending=False).reset_index(drop=True)
    corr_df.to_csv(outdir / "headwise_spearman.csv", index=False)

    with open(outdir / "run_metadata.json", "w") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "target_name": "recipient.cfscore",
                "num_records": len(records),
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "head_dim": HEAD_DIM,
                "ridge_alpha": alpha,
                "cv_splits": n_splits,
                "seed": seed,
            },
            f,
            indent=2,
        )

    return corr_df, corr


def fit_predict_for_head_cv(
    X_head: np.ndarray,
    y: np.ndarray,
    n_splits: int = 2,
    alpha: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    if len(y) < n_splits:
        return np.full_like(y, fill_value=np.nan, dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.full_like(y, fill_value=np.nan, dtype=np.float32)

    for train_idx, test_idx in kf.split(X_head):
        X_train, X_test = X_head[train_idx], X_head[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        preds[test_idx] = model.predict(X_test)

    return preds


def compute_topk_ensemble_predictions(
    records: Sequence[Record],
    corr_df: pd.DataFrame,
    K_values: Sequence[int],
    alpha: float = 1.0,
    n_splits: int = 2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[int, List[Tuple[int, int]]], np.ndarray]:
    X, y = flatten_features(records)

    per_head_preds: Dict[Tuple[int, int], np.ndarray] = {}
    for _, row in corr_df.iterrows():
        layer = int(row["layer"]) - 1
        head = int(row["head"]) - 1
        key = (layer, head)
        if key not in per_head_preds:
            X_head = X[:, layer, head, :]
            per_head_preds[key] = fit_predict_for_head_cv(
                X_head=X_head,
                y=y,
                n_splits=n_splits,
                alpha=alpha,
                seed=seed,
            )

    ensemble_rows = []
    ensemble_preds_dict: Dict[int, np.ndarray] = {}
    top_heads_dict: Dict[int, List[Tuple[int, int]]] = {}

    for K in K_values:
        topk = corr_df.head(K).copy()
        top_heads = [(int(r["layer"]) - 1, int(r["head"]) - 1) for _, r in topk.iterrows()]
        top_heads_dict[K] = top_heads

        stacked = np.stack([per_head_preds[h] for h in top_heads], axis=0)  # [K, N]
        y_pred = np.nanmean(stacked, axis=0)
        ensemble_preds_dict[K] = y_pred

        mask = ~np.isnan(y_pred)
        rho = np.nan
        if mask.sum() >= 3:
            rho, _ = spearmanr(y[mask], y_pred[mask])

        ensemble_rows.append(
            {
                "K": int(K),
                "spearman_cv_ensemble": float(rho) if not np.isnan(rho) else np.nan,
            }
        )

    ensemble_df = pd.DataFrame(ensemble_rows)
    return ensemble_df, ensemble_preds_dict, top_heads_dict, y


def plot_figure3_style_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    K: int,
    outpath: Path,
) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(y_pred, y_true, alpha=0.8)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"recipient.cfscore vs Top-{K} ensemble prediction")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def save_topk_outputs(
    ensemble_df: pd.DataFrame,
    ensemble_preds_dict: Dict[int, np.ndarray],
    top_heads_dict: Dict[int, List[Tuple[int, int]]],
    y_true: np.ndarray,
    records: Sequence[Record],
    outdir: Path,
) -> None:
    ensemble_df.to_csv(outdir / "topk_ensemble_results.csv", index=False)

    rows = []
    for K, heads in top_heads_dict.items():
        for rank, (layer, head) in enumerate(heads, start=1):
            rows.append(
                {
                    "K": int(K),
                    "rank_within_K": int(rank),
                    "layer": int(layer + 1),
                    "head": int(head + 1),
                }
            )
    pd.DataFrame(rows).to_csv(outdir / "topk_selected_heads.csv", index=False)

    senator_names = [rec.senator for rec in records]

    pred_df = pd.DataFrame({
        "senator": senator_names,
        "y_true": y_true,
    })
    for K, preds in ensemble_preds_dict.items():
        pred_df[f"y_pred_top{K}"] = preds
    pred_df.to_csv(outdir / "topk_ensemble_predictions.csv", index=False)


def plot_heatmap(corr: np.ndarray, outpath: Path) -> None:
    plt.figure(figsize=(12, 9))
    im = plt.imshow(corr, aspect="auto", origin="lower")
    plt.colorbar(im, label="Correlation")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.xticks(ticks=np.arange(NUM_HEADS), labels=np.arange(1, NUM_HEADS + 1))
    plt.yticks(ticks=np.arange(NUM_LAYERS), labels=np.arange(1, NUM_LAYERS + 1))
    #plt.title("Mistral attention-head probe performance for recipient.cfscore")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_topk_curve(ensemble_df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(
        ensemble_df["K"],
        ensemble_df["spearman_cv_ensemble"],
        marker="o"
    )
    plt.xlabel("Number of top heads (K)")
    plt.ylabel("Correlation")
    #plt.title("Top-K ensemble performance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def save_summary(corr_df: pd.DataFrame, first5_df: pd.DataFrame, outdir: Path) -> None:
    top10 = corr_df.head(10).copy()
    lines = []
    lines.append("Pipeline summary")
    lines.append("================")
    lines.append("")
    lines.append("Target: recipient.cfscore")
    lines.append("")
    lines.append("First 5 senators used for prompt record:")
    for _, row in first5_df.iterrows():
        lines.append(f"- {row['senator']}: CFScore={row['cfscore']:.4f}")
    lines.append("")
    lines.append("Top 10 heads by 2-fold CV Spearman correlation:")
    for _, row in top10.iterrows():
        lines.append(
            f"- Layer {int(row['layer'])}, Head {int(row['head'])}: "
            f"rho={row['spearman_cv']:.4f}"
        )
    (outdir / "summary.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfs-csv",
        type=str,
        default="/home/pippalin2/probing_llm/data/116th_Senate_CFS.csv",
        help="Path to CFS CSV.",
    )
    parser.add_argument("--outdir", type=str, default="mistral_cfs_outputs")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--cv-splits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of senators, for debugging.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfs_df = load_cfs_data(Path(args.cfs_csv), first_n=args.limit)
    first5_df = save_first_five_prompts(cfs_df, outdir)

    print(f"Loaded {len(cfs_df)} lawmakers from {args.cfs_csv}")
    print("Target column: recipient.cfscore")
    print("First 5 prompt records saved to:", outdir / "first5_senator_prompts.csv")

    tokenizer, model = load_model(cache_dir=args.cache_dir)
    records = extract_records(cfs_df, tokenizer, model, outdir)
    print("Saved activation records to:", outdir / "all_records.pt")

    corr_df, corr = compute_headwise_correlations(
        records=records,
        outdir=outdir,
        alpha=args.ridge_alpha,
        n_splits=args.cv_splits,
        seed=args.seed,
    )
    plot_heatmap(corr, outdir / "headwise_spearman_heatmap.png")
    save_summary(corr_df, first5_df, outdir)

    K_values = [1, 8, 32, 64, 96]
    ensemble_df, ensemble_preds_dict, top_heads_dict, y_true = compute_topk_ensemble_predictions(
        records=records,
        corr_df=corr_df,
        K_values=K_values,
        alpha=args.ridge_alpha,
        n_splits=args.cv_splits,
        seed=args.seed,
    )

    plot_topk_curve(ensemble_df, outdir / "topk_curve.png")

    save_topk_outputs(
        ensemble_df=ensemble_df,
        ensemble_preds_dict=ensemble_preds_dict,
        top_heads_dict=top_heads_dict,
        y_true=y_true,
        records=records,
        outdir=outdir,
    )

    for K in K_values:
        plot_figure3_style_scatter(
            y_true=y_true,
            y_pred=ensemble_preds_dict[K],
            K=K,
            outpath=outdir / f"topk_scatter_K{K}.png",
        )

    print("Saved headwise correlations to:", outdir / "headwise_spearman.csv")
    print("Saved heatmap to:", outdir / "headwise_spearman_heatmap.png")
    print("Saved Top-K ensemble outputs to:", outdir / "topk_ensemble_results.csv")
    print("Done.")


if __name__ == "__main__":
    main()
