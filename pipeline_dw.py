"""
Pipeline for:
1) reading DW.csv,
2) recording prompts for the first 5 senators,
3) extracting per-layer and per-head activations from Mistral-7B-Instruct,
4) training ridge linear probes for DW-NOMINATE dimension 1,
5) computing Spearman correlation for every layer/head,
6) saving results and plotting a heatmap.

Usage example:
python mistral_dw_pipeline.py \
    --dw-csv /path/to/DW.csv \
    --outdir ./mistral_dw_outputs

Notes:
- This script follows the activation-capture pattern in the user's mistral_test.py:
  it hooks the INPUT to each layer's self_attn.o_proj and averages over tokens.
- It uses 2-fold cross-validation and ridge probes, following the linear-probe setup
  described in the attached political representation paper.
"""

from __future__ import annotations

import argparse
import json
import os
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
    dw1: float
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
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    # Fallback: first object/string column with many unique values
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        raise ValueError("Could not infer senator-name column in DW.csv.")
    return max(obj_cols, key=lambda c: df[c].nunique())


def infer_dw1_column(df: pd.DataFrame) -> str:
    exact_candidates = [
        "dw1",
        "dw_nominate_dim1",
        "dw_nominate_1",
        "nominate_dim1",
        "nominate1",
        "dim1",
        "dimension1",
        "dwnom1",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in exact_candidates:
        if c in lower_map:
            return lower_map[c]

    fuzzy = []
    for c in df.columns:
        lc = c.lower()
        if ("dw" in lc or "nominate" in lc) and ("1" in lc or "dim1" in lc or "dimension1" in lc):
            fuzzy.append(c)
    if fuzzy:
        return fuzzy[0]

    # final fallback: numeric column with values mostly in [-1.2, 1.2]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    bounded = []
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        if s.min() >= -1.2 and s.max() <= 1.2:
            bounded.append(c)
    if bounded:
        return bounded[0]

    raise ValueError("Could not infer DW-NOMINATE first-dimension column in DW.csv.")


def build_prompt(name: str, year_prefix: bool = True) -> str:
    # Closer to the prompt template used in the linear-representation paper.
    if year_prefix:
        return (
            f"USER: Generate a statement by {name}, a politician in the United States.\n"
            f"ASSISTANT: In 2019, {name} said that"
        )
    return f"Generate a statement by {name}, a politician in the United States."


def load_dw_data(dw_csv: Path, first_n: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(dw_csv)
    name_col = infer_name_column(df)
    dw1_col = infer_dw1_column(df)

    out = df[[name_col, dw1_col]].copy()
    out.columns = ["senator", "dw1"]
    out["senator"] = out["senator"].map(normalize_name)
    out["dw1"] = pd.to_numeric(out["dw1"], errors="coerce")
    out = out.dropna(subset=["senator", "dw1"]).drop_duplicates(subset=["senator"]).reset_index(drop=True)

    if first_n is not None:
        out = out.head(first_n).copy()

    return out


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
            dw1 = float(row["dw1"])
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
                    dw1=dw1,
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
        y[i] = rec.dw1
    return X, y


def cv_spearman_for_head(
    X_head: np.ndarray,
    y: np.ndarray,
    n_splits: int = 2,
    alpha: float = 1.0,
    seed: int = 42,
) -> float:
    # X_head: [N, D]
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

def compute_oof_predictions_all_heads(
    records: Sequence[Record],
    alpha: float = 1.0,
    n_splits: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Compute out-of-fold predictions for every head.
    Returns:
        y: [N]
        all_preds: [L, H, N]
        head_df: dataframe with layer/head/spearman_cv
    """
    X, y = flatten_features(records)  # [N, L, H, D], [N]
    N = X.shape[0]

    all_preds = np.full((NUM_LAYERS, NUM_HEADS, N), np.nan, dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        for layer in range(NUM_LAYERS):
            for head in range(NUM_HEADS):
                X_head = X[:, layer, head, :]  # [N, D]

                X_train = X_head[train_idx]
                X_test = X_head[test_idx]
                y_train = y[train_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = Ridge(alpha=alpha)
                model.fit(X_train, y_train)

                preds = model.predict(X_test)
                all_preds[layer, head, test_idx] = preds.astype(np.float32)

    rows = []
    for layer in range(NUM_LAYERS):
        for head in range(NUM_HEADS):
            mask = ~np.isnan(all_preds[layer, head])
            if mask.sum() < 3:
                rho = np.nan
            else:
                rho, _ = spearmanr(y[mask], all_preds[layer, head, mask])
            rows.append(
                {
                    "layer": layer + 1,
                    "head": head + 1,
                    "spearman_cv": float(rho),
                }
            )

    head_df = pd.DataFrame(rows).sort_values("spearman_cv", ascending=False).reset_index(drop=True)
    return y, all_preds, head_df


def compute_topk_ensemble_results(
    y: np.ndarray,
    all_preds: np.ndarray,
    head_df: pd.DataFrame,
    K_values: Sequence[int] = (1, 8, 32, 64, 96),
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[int, List[Tuple[int, int]]]]:
    """
    Compute ensemble predictions using top-K heads, like the paper.
    Returns:
        ensemble_df: one row per K with Spearman
        ensemble_preds_dict: K -> [N] predictions
        top_heads_dict: K -> list of (layer, head) using 1-indexed coordinates
    """
    ensemble_rows = []
    ensemble_preds_dict: Dict[int, np.ndarray] = {}
    top_heads_dict: Dict[int, List[Tuple[int, int]]] = {}

    valid_head_df = head_df.dropna(subset=["spearman_cv"]).copy()

    for K in K_values:
        topk = valid_head_df.head(K).copy()

        preds_list = []
        head_list = []

        for _, row in topk.iterrows():
            layer = int(row["layer"]) - 1
            head = int(row["head"]) - 1
            preds_list.append(all_preds[layer, head])   # [N]
            head_list.append((layer + 1, head + 1))

        ensemble_preds = np.mean(np.stack(preds_list, axis=0), axis=0)
        rho, _ = spearmanr(y, ensemble_preds)

        ensemble_rows.append(
            {
                "K": int(K),
                "spearman_cv_ensemble": float(rho),
            }
        )
        ensemble_preds_dict[int(K)] = ensemble_preds
        top_heads_dict[int(K)] = head_list

    ensemble_df = pd.DataFrame(ensemble_rows).sort_values("K").reset_index(drop=True)
    return ensemble_df, ensemble_preds_dict, top_heads_dict


def plot_topk_curve(ensemble_df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(
        ensemble_df["K"],
        ensemble_df["spearman_cv_ensemble"],
        marker="o",
    )
    plt.xlabel("Number of top heads (K)")
    plt.ylabel("Correlation")
    #plt.title("Top-K ensemble performance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_figure3_style_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    K: int,
    rho: float,
    outpath: Path,
) -> None:
    plt.figure(figsize=(7, 7))
    
    sc = plt.scatter(
        y_pred,
        y_true,
        c=y_true,                 # color by true DW score
        cmap="bwr",               # blue-white-red
        vmin=-1.0,
        vmax=1.0,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.6,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.text(
        0.98,
        0.02,
        rf"$\rho^{{CV}}_{{K={K}}} = {rho:.3f}$",
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        fontsize=14,
    )
    cbar = plt.colorbar(sc)
    #cbar.set_label("True DW-NOMINATE score")
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
                    "layer": int(layer),
                    "head": int(head),
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
    #plt.title("Mistral attention-head probe performance for DW-NOMINATE dimension 1")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary(corr_df: pd.DataFrame, first5_df: pd.DataFrame, outdir: Path) -> None:
    top10 = corr_df.head(10).copy()
    lines = []
    lines.append("Pipeline summary")
    lines.append("================")
    lines.append("")
    lines.append("First 5 senators used for prompt record:")
    for _, row in first5_df.iterrows():
        lines.append(f"- {row['senator']}: DW1={row['dw1']:.4f}")
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
    parser.add_argument("--dw-csv",
    type=str,
    default="/home/pippalin2/probing_llm/data/116th_Senate_DW.csv",
    help="Path to DW.csv")
    parser.add_argument("--outdir", type=str, default="mistral_dw_outputs")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--cv-splits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=102, # how many senators
        help="Optional limit on number of senators, for debugging.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dw_df = load_dw_data(Path(args.dw_csv), first_n=args.limit)
    first5_df = save_first_five_prompts(dw_df, outdir)

    print(f"Loaded {len(dw_df)} lawmakers from {args.dw_csv}")
    print("First 5 prompt records saved to:", outdir / "first5_senator_prompts.csv")

    tokenizer, model = load_model(cache_dir=args.cache_dir)
    records = extract_records(dw_df, tokenizer, model, outdir)
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
        # ---- Top-K ensemble results (paper-style) ----
    K_values = [1, 8, 32, 64, 96]

    y_true, all_preds, head_df = compute_oof_predictions_all_heads(
        records=records,
        alpha=args.ridge_alpha,
        n_splits=args.cv_splits,
        seed=args.seed,
    )

    ensemble_df, ensemble_preds_dict, top_heads_dict = compute_topk_ensemble_results(
        y=y_true,
        all_preds=all_preds,
        head_df=head_df,
        K_values=K_values,
    )

    save_topk_outputs(
        ensemble_df=ensemble_df,
        ensemble_preds_dict=ensemble_preds_dict,
        top_heads_dict=top_heads_dict,
        y_true=y_true,
        records=records,
        outdir=outdir,)

    plot_topk_curve(ensemble_df, outdir / "topk_ensemble_curve.png")

    # Figure-3-style scatter for K=32
    # Figure-3-style scatter for K=32
    if 32 in ensemble_preds_dict:
        rho32 = float(
            ensemble_df.loc[ensemble_df["K"] == 32, "spearman_cv_ensemble"].iloc[0]
        )
        plot_figure3_style_scatter(
            y_true=y_true,
            y_pred=ensemble_preds_dict[32],
            K=32,
            rho=rho32,
            outpath=outdir / "figure3_style_top32_scatter.png",
        )

    print("Saved top-K ensemble results to:", outdir / "topk_ensemble_results.csv")
    print("Saved top-K curve to:", outdir / "topk_ensemble_curve.png")


    print("Saved headwise correlations to:", outdir / "headwise_spearman.csv")
    print("Saved heatmap to:", outdir / "headwise_spearman_heatmap.png")
    print("Done.")


if __name__ == "__main__":
    main()
