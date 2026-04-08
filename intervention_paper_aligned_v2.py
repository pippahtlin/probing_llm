from __future__ import annotations

import argparse
import json
import os
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Paper-aligned intervention runner for Mistral-7B-Instruct.
# Aligned to the probe-training script by:
#   1) hooking the INPUT to each layer's self_attn.o_proj
#   2) using the same top-k head definitions from training
#   3) using top-k = 32 by default
#   4) scoring generated tokens only by default
#   5) estimating sigma from head-vector norms over training records
#   6) scoring activations ONLINE during generation instead of
#      rescoring the final text in a separate forward pass
# ============================================================

BASE_DIR = Path(os.path.expanduser("~/probing_llm"))
DW_DIR = BASE_DIR / "mistral_dw_outputs"
CFS_DIR = BASE_DIR / "mistral_cfs_outputs"
BT_DIR = BASE_DIR / "mistral_bt_outputs"
DATA_DIR = BASE_DIR / "data"

DEFAULT_DW_ARTIFACTS = DW_DIR / "probe_artifacts.pt"
DEFAULT_CFS_ARTIFACTS = CFS_DIR / "probe_artifacts.pt"
DEFAULT_BT_ARTIFACTS = BT_DIR / "probe_artifacts.pt"

DEFAULT_DW_TOPK_CSV = DW_DIR / "topk_selected_heads.csv"
DEFAULT_CFS_TOPK_CSV = CFS_DIR / "topk_selected_heads.csv"
DEFAULT_BT_TOPK_CSV = BT_DIR / "topk_selected_heads.csv"

DEFAULT_DW_RECORDS = DW_DIR / "all_records.pt"
DEFAULT_CFS_RECORDS = CFS_DIR / "all_records.pt"
DEFAULT_BT_RECORDS = BT_DIR / "all_records.pt"

DEFAULT_DW_CSV = DATA_DIR / "116th_Senate_DW.csv"
DEFAULT_CFS_CSV = DATA_DIR / "116th_Senate_CFS.csv"
DEFAULT_BT_CSV = DATA_DIR / "gpt_bt_score.csv"

DEFAULT_OUTDIR = BASE_DIR / "intervention_outputs_v2"

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
    layer_hidden_avg: torch.Tensor
    head_avg: torch.Tensor


@dataclass
class ProbeInfo:
    alpha: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_name(name: str) -> str:
    name = str(name).strip()
    return re.sub(r"\s+", " ", name)


def infer_name_column(df: pd.DataFrame) -> str:
    candidates = [
        "senator", "name", "full_name", "bioname",
        "member_name", "person_name", "legislator", "lawmaker"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        raise ValueError("Could not infer name column.")
    return max(obj_cols, key=lambda c: df[c].nunique())


def infer_target_column(df: pd.DataFrame, score_type: str, explicit_col: Optional[str] = None) -> str:
    if explicit_col is not None:
        if explicit_col not in df.columns:
            raise ValueError(f"Column {explicit_col!r} not found in CSV.")
        return explicit_col

    lower_map = {c.lower(): c for c in df.columns}

    if score_type == "dw":
        candidates = [
            "dw1", "dw_nominate_dim1", "dw_nominate_1",
            "nominate_dim1", "nominate1", "dim1", "dimension1", "dwnom1",
        ]
    elif score_type == "cfs":
        candidates = ["cfscore", "recipient_cfscore", "cf_score", "cfs"]
    elif score_type == "bt":
        candidates = ["bt", "bt_score", "bradley_terry", "bradley_terry_score", "lamp", "score"]
    else:
        raise ValueError(f"Unknown score type: {score_type}")

    for c in candidates:
        if c in lower_map:
            return lower_map[c]

    fuzzy = []
    for c in df.columns:
        lc = c.lower()
        if score_type == "dw" and ("dw" in lc or "nominate" in lc):
            fuzzy.append(c)
        elif score_type == "cfs" and ("cf" in lc or "cfs" in lc or "bonica" in lc):
            fuzzy.append(c)
        elif score_type == "bt" and ("bt" in lc or "bradley" in lc or "lamp" in lc):
            fuzzy.append(c)
    if fuzzy:
        return fuzzy[0]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError(f"Could not infer target column for {score_type}.")
    return numeric_cols[0]


def load_target_stats(csv_path: Path, score_type: str, explicit_col: Optional[str] = None) -> Dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV for {score_type} not found: {csv_path}")
    df = pd.read_csv(csv_path)
    name_col = infer_name_column(df)
    target_col = infer_target_column(df, score_type, explicit_col)

    out = df[[name_col, target_col]].copy()
    out.columns = ["name", "score"]
    out["name"] = out["name"].map(normalize_name)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["score"])

    s = out["score"]
    return {
        "name_col": name_col,
        "target_col": target_col,
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
    }


def normalize_score(score: float, stats: Dict[str, float], score_type: str) -> float:
    lo, hi = stats["min"], stats["max"]
    if lo >= -1.05 and hi <= 1.05:
        return float(score)
    if hi <= lo:
        return 0.0
    return float(2.0 * (score - lo) / (hi - lo) - 1.0)


def _key_to_tuple(key) -> Tuple[int, int]:
    if isinstance(key, tuple):
        return int(key[0]), int(key[1])
    if isinstance(key, str):
        nums = [int(x) for x in re.findall(r"\d+", key)]
        if len(nums) >= 2:
            return nums[0], nums[1]
    raise ValueError(f"Cannot parse probe key: {key!r}")


def load_probe_artifacts(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Probe artifact not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a dict in {path}, got {type(obj)}")
    return obj


def get_probe_info(artifact: Dict, layer_1idx: int, head_1idx: int) -> ProbeInfo:
    probes = artifact["probes"]
    probe = probes.get((layer_1idx, head_1idx))
    if probe is None:
        for k, v in probes.items():
            lk, hk = _key_to_tuple(k)
            if lk == layer_1idx and hk == head_1idx:
                probe = v
                break
    if probe is None:
        raise KeyError(f"Probe for layer={layer_1idx}, head={head_1idx} not found.")

    return ProbeInfo(
        alpha=float(probe["alpha"]),
        scaler_mean=np.asarray(probe["scaler_mean"], dtype=np.float32),
        scaler_scale=np.asarray(probe["scaler_scale"], dtype=np.float32),
        coef=np.asarray(probe["coef"], dtype=np.float32),
        intercept=float(np.asarray(probe["intercept"]).reshape(-1)[0]),
    )


def load_top_heads_from_csv(csv_path: Path, k: int) -> List[Tuple[int, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Top-head CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "K" in df.columns:
        df = df[df["K"] == k].copy()
    if "rank_within_K" in df.columns:
        df = df.sort_values("rank_within_K")
    if not {"layer", "head"}.issubset(df.columns):
        raise ValueError(f"Expected columns layer/head in {csv_path}")
    pairs = list(zip(df["layer"].astype(int), df["head"].astype(int)))
    if len(pairs) < k:
        raise ValueError(f"Only found {len(pairs)} heads in {csv_path}, need {k}")
    return pairs[:k]


def predict_with_probe(head_vec: np.ndarray, probe: ProbeInfo) -> float:
    x = (head_vec.astype(np.float32) - probe.scaler_mean) / np.maximum(probe.scaler_scale, 1e-8)
    return float(np.dot(probe.coef, x) + probe.intercept)


def average_probe_prediction(
    head_avg: np.ndarray,
    artifact: Dict,
    top_heads: Sequence[Tuple[int, int]],
    score_type: str,
    stats: Dict[str, float],
) -> float:
    vals = []
    for layer_1idx, head_1idx in top_heads:
        probe = get_probe_info(artifact, layer_1idx, head_1idx)
        vec = head_avg[layer_1idx - 1, head_1idx - 1]
        pred = predict_with_probe(vec, probe)
        vals.append(normalize_score(pred, stats, score_type))
    return float(np.mean(vals)) if vals else np.nan


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


def build_generation_prompt(topic: str) -> str:
    return f"USER: Write a statement about {topic}.\nASSISTANT: Regarding {topic}, I believe that"


def load_records(path: Path) -> List[Record]:
    if not path.exists():
        raise FileNotFoundError(f"all_records.pt not found: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, list):
        raise ValueError(f"Expected list in {path}, got {type(obj)}")
    return obj


def estimate_head_sigmas(records: Sequence[Record]) -> np.ndarray:
    """
    Estimate sigma_hat_{l,h} from training activations using the std of head-vector norms.
    This is closer to a scalar scale for the full head activation vector than averaging
    per-dimension stds.
    """
    if len(records) == 0:
        raise ValueError("No records found for sigma estimation.")
    X = np.stack([
        rec.head_avg.numpy() if isinstance(rec.head_avg, torch.Tensor) else np.asarray(rec.head_avg)
        for rec in records
    ], axis=0)  # [N, L, H, D]
    norms = np.linalg.norm(X, axis=-1)      # [N, L, H]
    sigma = norms.std(axis=0, ddof=0)       # [L, H]
    return np.maximum(sigma, 1e-8).astype(np.float32)


def make_delta_from_probe(probe: ProbeInfo, sigma_hat_lh: float, alpha: float) -> np.ndarray:
    raw_direction = probe.coef / np.maximum(probe.scaler_scale, 1e-8)
    return (alpha * float(sigma_hat_lh) * raw_direction).astype(np.float32)


class GenerationCaptureIntervention:
    """
    Single hook manager that both:
      1) applies intervention at the exact activation definition used in training
         (input to self_attn.o_proj), and
      2) accumulates token-wise online activations for generated tokens only.

    For cached generation:
      - first forward pass processes the full prompt (seq len = prompt_len)
      - each later forward pass processes exactly one new token (seq len = 1)
    We skip accumulation on the prompt pass by default.
    """
    def __init__(
        self,
        model,
        source_artifact: Dict,
        heads_to_use: Sequence[Tuple[int, int]],
        alpha: float,
        sigma_hat: np.ndarray,
        score_generated_only: bool = True,
    ):
        self.model = model
        self.source_artifact = source_artifact
        self.heads_to_use = list(heads_to_use)
        self.alpha = alpha
        self.sigma_hat = sigma_hat
        self.score_generated_only = score_generated_only
        self.hooks = []
        self.call_count = 0
        self.sum_by_layer = np.zeros((NUM_LAYERS, NUM_HEADS, HEAD_DIM), dtype=np.float64)
        self.count_by_layer = np.zeros(NUM_LAYERS, dtype=np.int64)

        self.layer_to_head_delta: Dict[int, Dict[int, np.ndarray]] = {}
        for layer_1idx, head_1idx in self.heads_to_use:
            probe = get_probe_info(source_artifact, layer_1idx, head_1idx)
            sigma_lh = float(self.sigma_hat[layer_1idx - 1, head_1idx - 1])
            delta = make_delta_from_probe(probe, sigma_hat_lh=sigma_lh, alpha=alpha)
            self.layer_to_head_delta.setdefault(layer_1idx - 1, {})[head_1idx - 1] = delta

    def __enter__(self):
        for layer_idx in range(NUM_LAYERS):
            per_head = self.layer_to_head_delta.get(layer_idx, {})

            def make_hook(layer_idx=layer_idx, per_head=per_head):
                def hook(module, inputs):
                    x = inputs[0]  # [B, T, H]
                    bsz, tsz, hsz = x.shape
                    x_view = x.view(bsz, tsz, NUM_HEADS, HEAD_DIM).clone()

                    # Apply intervention on all tokens seen in this forward call.
                    for head_idx, delta_np in per_head.items():
                        delta = torch.as_tensor(delta_np, device=x.device, dtype=x.dtype)
                        x_view[:, :, head_idx, :] = x_view[:, :, head_idx, :] + delta.view(1, 1, -1)

                    # Online accumulation:
                    # skip the first call (prompt pass) if score_generated_only=True.
                    keep_this_call = True
                    if self.score_generated_only and self.call_count == 0:
                        keep_this_call = False

                    if keep_this_call:
                        x_np = x_view.detach().float().cpu().numpy()   # [B, T, Hh, D]
                        token_sum = x_np.sum(axis=(0, 1))              # [Hh, D]
                        token_count = x_np.shape[0] * x_np.shape[1]
                        self.sum_by_layer[layer_idx] += token_sum
                        self.count_by_layer[layer_idx] += token_count

                    return (x_view.view(bsz, tsz, hsz),)
                return hook

            h = self.model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(make_hook())
            self.hooks.append(h)

        return self

    def note_forward_completed(self):
        self.call_count += 1

    def get_head_average(self) -> np.ndarray:
        out = np.zeros((NUM_LAYERS, NUM_HEADS, HEAD_DIM), dtype=np.float32)
        for layer_idx in range(NUM_LAYERS):
            cnt = int(self.count_by_layer[layer_idx])
            if cnt > 0:
                out[layer_idx] = (self.sum_by_layer[layer_idx] / cnt).astype(np.float32)
        return out

    def __exit__(self, exc_type, exc, tb):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, next_sorted)

    return torch.multinomial(probs, num_samples=1)


def generate_one_online(
    model,
    tokenizer,
    prompt: str,
    seed: int,
    capture_manager: GenerationCaptureIntervention,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.95,
):
    set_all_seeds(seed)
    device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = int(input_ids.shape[1])

    generated = input_ids.clone()
    attn = attention_mask.clone()
    past_key_values = None

    for step in range(max_new_tokens):
        if step == 0:
            model_inputs = {
                "input_ids": generated,
                "attention_mask": attn,
                "use_cache": True,
            }
        else:
            model_inputs = {
                "input_ids": generated[:, -1:],
                "attention_mask": attn,
                "past_key_values": past_key_values,
                "use_cache": True,
            }

        with torch.no_grad():
            outputs = model(**model_inputs)
        capture_manager.note_forward_completed()

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = sample_next_token(logits, temperature=temperature, top_p=top_p)

        generated = torch.cat([generated, next_token], dim=1)
        attn = torch.cat([attn, torch.ones((attn.shape[0], 1), dtype=attn.dtype, device=device)], dim=1)

        if int(next_token.item()) == tokenizer.eos_token_id:
            break

    output_ids = generated.detach().cpu()
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_ids = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "full_text": output_text,
        "generated_text": generated_text,
        "output_ids": output_ids,
        "prompt_len": prompt_len,
    }


def sample_random_heads(k: int, seed: int) -> List[Tuple[int, int]]:
    all_heads = [(l + 1, h + 1) for l in range(NUM_LAYERS) for h in range(NUM_HEADS)]
    rng = random.Random(seed)
    rng.shuffle(all_heads)
    return all_heads[:k]


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dw-artifacts", type=str, default=str(DEFAULT_DW_ARTIFACTS))
    parser.add_argument("--cfs-artifacts", type=str, default=str(DEFAULT_CFS_ARTIFACTS))
    parser.add_argument("--bt-artifacts", type=str, default=str(DEFAULT_BT_ARTIFACTS))

    parser.add_argument("--dw-topk-csv", type=str, default=str(DEFAULT_DW_TOPK_CSV))
    parser.add_argument("--cfs-topk-csv", type=str, default=str(DEFAULT_CFS_TOPK_CSV))
    parser.add_argument("--bt-topk-csv", type=str, default=str(DEFAULT_BT_TOPK_CSV))

    parser.add_argument("--dw-records", type=str, default=str(DEFAULT_DW_RECORDS))
    parser.add_argument("--cfs-records", type=str, default=str(DEFAULT_CFS_RECORDS))
    parser.add_argument("--bt-records", type=str, default=str(DEFAULT_BT_RECORDS))

    parser.add_argument("--dw-csv", type=str, default=str(DEFAULT_DW_CSV))
    parser.add_argument("--cfs-csv", type=str, default=str(DEFAULT_CFS_CSV))
    parser.add_argument("--bt-csv", type=str, default=str(DEFAULT_BT_CSV))

    parser.add_argument("--dw-col", type=str, default=None)
    parser.add_argument("--cfs-col", type=str, default=None)
    parser.add_argument("--bt-col", type=str, default=None)

    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)

    parser.add_argument("--source-scores", nargs="+", default=["dw", "cfs", "bt"])
    parser.add_argument("--topics", nargs="+", default=["immigration", "abortion", "gun control", "climate change", "lgbtq rights"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[-20, -10, 0, 10, 20])

    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n-generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)

    # Default matches paper-style generated-token scoring.
    parser.add_argument("--include-prompt-in-score", action="store_true", default=False)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / "texts")

    artifacts = {
        "dw": load_probe_artifacts(Path(args.dw_artifacts)),
        "cfs": load_probe_artifacts(Path(args.cfs_artifacts)),
        "bt": load_probe_artifacts(Path(args.bt_artifacts)),
    }
    stats = {
        "dw": load_target_stats(Path(args.dw_csv), "dw", args.dw_col),
        "cfs": load_target_stats(Path(args.cfs_csv), "cfs", args.cfs_col),
        "bt": load_target_stats(Path(args.bt_csv), "bt", args.bt_col),
    }
    top_heads = {
        "dw": load_top_heads_from_csv(Path(args.dw_topk_csv), args.k),
        "cfs": load_top_heads_from_csv(Path(args.cfs_topk_csv), args.k),
        "bt": load_top_heads_from_csv(Path(args.bt_topk_csv), args.k),
    }
    random_heads = {
        score: sample_random_heads(args.k, args.seed + 1009 * i)
        for i, score in enumerate(["dw", "cfs", "bt"])
    }
    sigma_hat = {
        "dw": estimate_head_sigmas(load_records(Path(args.dw_records))),
        "cfs": estimate_head_sigmas(load_records(Path(args.cfs_records))),
        "bt": estimate_head_sigmas(load_records(Path(args.bt_records))),
    }

    save_json(
        {
            "model_name": MODEL_NAME,
            "base_dir": str(BASE_DIR),
            "outdir": str(outdir),
            "source_scores": args.source_scores,
            "topics": args.topics,
            "alphas": args.alphas,
            "k": args.k,
            "n_generations": args.n_generations,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "score_generated_only": not args.include_prompt_in_score,
            "sigma_definition": "std_of_head_vector_norms_over_training_records",
            "top_heads": {k: v for k, v in top_heads.items()},
            "random_heads": {k: v for k, v in random_heads.items()},
            "csv_stats": stats,
        },
        outdir / "run_metadata.json",
    )

    print("Loading model...")
    tokenizer, model = load_model(cache_dir=args.cache_dir)

    per_generation_rows = []
    aggregate_rows = []

    for source_score in args.source_scores:
        source_artifact = artifacts[source_score]
        source_top_heads = top_heads[source_score]
        source_random_heads = random_heads[source_score]
        source_sigma_hat = sigma_hat[source_score]

        for head_mode, heads_for_intervention in [("topk", source_top_heads), ("random", source_random_heads)]:
            for topic in args.topics:
                prompt = build_generation_prompt(topic)

                for alpha in args.alphas:
                    print(f"\n=== source={source_score} mode={head_mode} topic={topic} alpha={alpha} ===")
                    one_setting_rows = []

                    for gen_idx in range(args.n_generations):
                        # Keep the same seed across alphas for a fixed generation index.
                        seed = args.seed + gen_idx

                        with GenerationCaptureIntervention(
                            model=model,
                            source_artifact=source_artifact,
                            heads_to_use=heads_for_intervention,
                            alpha=alpha,
                            sigma_hat=source_sigma_hat,
                            score_generated_only=not args.include_prompt_in_score,
                        ) as manager:
                            gen_out = generate_one_online(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=prompt,
                                seed=seed,
                                capture_manager=manager,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                            )
                            head_avg = manager.get_head_average()

                        dw_pred = average_probe_prediction(
                            head_avg=head_avg,
                            artifact=artifacts["dw"],
                            top_heads=top_heads["dw"],
                            score_type="dw",
                            stats=stats["dw"],
                        )
                        cfs_pred = average_probe_prediction(
                            head_avg=head_avg,
                            artifact=artifacts["cfs"],
                            top_heads=top_heads["cfs"],
                            score_type="cfs",
                            stats=stats["cfs"],
                        )
                        bt_pred = average_probe_prediction(
                            head_avg=head_avg,
                            artifact=artifacts["bt"],
                            top_heads=top_heads["bt"],
                            score_type="bt",
                            stats=stats["bt"],
                        )

                        row = {
                            "source_score": source_score,
                            "head_mode": head_mode,
                            "topic": topic,
                            "alpha": alpha,
                            "generation_idx": gen_idx + 1,
                            "seed": seed,
                            "dw_pred": dw_pred,
                            "cfs_pred": cfs_pred,
                            "bt_pred": bt_pred,
                            "prompt": prompt,
                            "generated_text": gen_out["generated_text"],
                            "full_text": gen_out["full_text"],
                        }
                        one_setting_rows.append(row)
                        per_generation_rows.append(row)

                        text_dir = outdir / "texts" / source_score / head_mode / topic.replace(" ", "_") / f"alpha_{alpha}"
                        ensure_dir(text_dir)
                        (text_dir / f"gen_{gen_idx+1}.txt").write_text(gen_out["full_text"])

                    agg = {
                        "source_score": source_score,
                        "head_mode": head_mode,
                        "topic": topic,
                        "alpha": alpha,
                        "n_generations": len(one_setting_rows),
                        "dw_pred_mean": float(np.mean([r["dw_pred"] for r in one_setting_rows])),
                        "cfs_pred_mean": float(np.mean([r["cfs_pred"] for r in one_setting_rows])),
                        "bt_pred_mean": float(np.mean([r["bt_pred"] for r in one_setting_rows])),
                        "dw_pred_std": float(np.std([r["dw_pred"] for r in one_setting_rows], ddof=0)),
                        "cfs_pred_std": float(np.std([r["cfs_pred"] for r in one_setting_rows], ddof=0)),
                        "bt_pred_std": float(np.std([r["bt_pred"] for r in one_setting_rows], ddof=0)),
                    }
                    aggregate_rows.append(agg)

    per_gen_df = pd.DataFrame(per_generation_rows)
    agg_df = pd.DataFrame(aggregate_rows)

    per_gen_df.to_csv(outdir / "per_generation_scores.csv", index=False)
    agg_df.to_csv(outdir / "aggregate_scores.csv", index=False)

    print("\nDone.")
    print("Saved:", outdir / "per_generation_scores.csv")
    print("Saved:", outdir / "aggregate_scores.csv")
    print("Saved texts under:", outdir / "texts")


if __name__ == "__main__":
    main()
