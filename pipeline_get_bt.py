#!/usr/bin/env python3
"""Estimate liberal-conservative Bradley-Terry scores for 116th U.S. Senators
using pairwise comparisons from an instruction-tuned Mistral model.

This script mirrors the Wu et al. workflow as closely as possible for the
liberal-conservative dimension:

1. Build all unique senator matchups.
2. Use the paper's asymmetric prompting rule:
   - D v D and D v R: ask which senator is more liberal.
   - R v R: ask which senator is more conservative.
3. Run a second extraction prompt to convert the free-form answer into either
   senator_1, senator_2, or Tie.
4. Treat ties as 0.5 wins for both senators.
5. Fit a Bradley-Terry model to the aggregated pairwise outcomes.
6. Return log-ability (lambda) scores and unit-interval rescaled scores.

Important note:
The original paper used the bias-reduced MLE from the BradleyTerry2 R package.
This script uses a standard MM / iterative scaling estimator for the classical
Bradley-Terry model with tie handling via 0.5 wins each, which is the same tie
coding rule described in the paper. If you need exact paper-matching bias
reduction, export the aggregated matchup table and fit in R with BradleyTerry2.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PARTY_MAP = {100: "D", 200: "R", 328: "I"}

STATE_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_ascii(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))

def canonicalize_text(text: str) -> str:
    text = normalize_ascii(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return clean_spaces(text)

def smart_title_token(tok: str) -> str:
    if tok.lower() in SUFFIXES:
        return tok.upper() if tok.lower() in {"ii", "iii", "iv", "v"} else tok.capitalize()
    return tok.capitalize()

def format_name_last_first(raw_name: str) -> str:
    raw_name = clean_spaces(str(raw_name))
    if "," in raw_name:
        last, first = [clean_spaces(x) for x in raw_name.split(",", 1)]
        return clean_spaces(" ".join([*(smart_title_token(t) for t in first.split()),
                                      *(smart_title_token(t) for t in last.split())]))
    return clean_spaces(" ".join(smart_title_token(t) for t in raw_name.split()))

def make_aliases(full_name: str) -> List[str]:
    parts = canonicalize_text(full_name).split()
    aliases = set()
    if parts:
        aliases.add(" ".join(parts))
        aliases.add(parts[-1])
        if len(parts) >= 2:
            aliases.add(f"{parts[0]} {parts[-1]}")
    return sorted(aliases, key=len, reverse=True)

@dataclass(frozen=True)
class Senator:
    idx: int
    full_name: str
    party: str
    state: str

    @property
    def label(self) -> str:
        return f"{self.full_name} ({self.party}-{self.state})"

def infer_state_abbrev(value: str) -> str:
    value = clean_spaces(str(value))
    if len(value) == 2 and value.isalpha():
        return value.upper()
    return STATE_TO_ABBR.get(value, value.upper())

def load_senators(csv_path: str) -> List[Senator]:
    df = pd.read_csv(csv_path)
    required = {"name", "party", "state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns {sorted(required)}. Missing: {sorted(missing)}")

    senators = []
    for i, row in df.reset_index(drop=True).iterrows():
        party_val = row["party"]
        party = PARTY_MAP.get(int(party_val), str(party_val)) if pd.notna(party_val) else "?"
        senators.append(Senator(
            idx=i,
            full_name=format_name_last_first(str(row["name"])),
            party=party,
            state=infer_state_abbrev(row["state"]),
        ))

    seen, deduped = set(), []
    for s in senators:
        key = (s.full_name, s.party, s.state)
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped

def comparison_prompt(s1: Senator, s2: Senator) -> Tuple[str, str]:
    if s1.party == "R" and s2.party == "R":
        return "conservative_prompt", (
            f"Which senator is more conservative: "
            f"{s1.label} or {s2.label}?\n"
            f"Answer with ONLY one of the following:\n"
            f"1. {s1.full_name}\n"
            f"2. {s2.full_name}\n"
            f"3. Tie"
        )
    return "liberal_prompt", (
        f"Which senator is more liberal: "
        f"{s1.label} or {s2.label}?\n"
        f"Answer with ONLY one of the following:\n"
        f"1. {s1.full_name}\n"
        f"2. {s2.full_name}\n"
        f"3. Tie"
    )

def extraction_prompt(raw_answer: str, s1: Senator, s2: Senator, orientation: str) -> str:
    if orientation == "conservative_prompt":
        extractor = (
            f"In the above Text, who is described to be the more conservative or less liberal senator: "
            f"{s1.full_name} or {s2.full_name}? Return only the full name without party affiliation "
            f"or state information. If one senator is described as more liberal, return the other senator's "
            f"name. If one senator is described as more moderate, return the other senator's name. "
            f"If neither senators are described to be more conservative, less liberal, more liberal, "
            f"or more moderate, reply with 'Tie.'"
        )
    else:
        extractor = (
            f"In the above Text, who is described to be the more liberal, more progressive, or less conservative senator: "
            f"{s1.full_name} or {s2.full_name}? Return only the full name without party affiliation or state information. "
            f"If one senator is described as more conservative, return the other senator's name. "
            f"If one senator is described as more moderate, return the other senator's name. "
            f"If neither senators are described to be more liberal, more progressive, less conservative, more conservative, "
            f"or more moderate, reply with 'Tie.'"
        )
    return raw_answer.rstrip() + "\n\n" + extractor

class MistralJudge:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: str = "auto",
        max_new_tokens: int = 30,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

        torch_dtype = {"auto": None, "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = dict(device_map="auto" if self.device == "cuda" else None)
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if self.device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

def parse_extracted_name(text: str, s1: Senator, s2: Senator) -> str:
    canon = canonicalize_text(clean_spaces(text))
    if not canon:
        return "Tie"
    if canon in {"tie", "a tie", "neither", "cannot determine", "can t determine"}:
        return "Tie"

    alias_map = {}
    for a in make_aliases(s1.full_name):
        alias_map[a] = s1.full_name
    for a in make_aliases(s2.full_name):
        alias_map[a] = s2.full_name

    if canon in alias_map:
        return alias_map[canon]

    hits = []
    for alias, full in alias_map.items():
        if re.search(rf"\b{re.escape(alias)}\b", canon):
            hits.append((len(alias), full))
    if hits:
        hits.sort(reverse=True)
        if len(hits) > 1 and hits[0][0] == hits[1][0] and hits[0][1] != hits[1][1]:
            return "Tie"
        return hits[0][1]
    return "Tie"

def adjudicate_pair(judge: MistralJudge, s1: Senator, s2: Senator) -> Dict[str, str]:
    orientation, cmp_prompt = comparison_prompt(s1, s2)
    cmp_answer = judge.generate(cmp_prompt)

    # ALWAYS use extractor
    ext_prompt = extraction_prompt(cmp_answer, s1, s2, orientation)
    ext_answer = judge.generate(ext_prompt)
    extracted = parse_extracted_name(ext_answer, s1, s2)

    if extracted == "Tie":
        winner = "Tie"
    elif orientation == "conservative_prompt":
        winner = extracted
    else:
        winner = s2.full_name if extracted == s1.full_name else s1.full_name

    return {
        "senator_1": s1.full_name,
        "party_1": s1.party,
        "state_1": s1.state,
        "senator_2": s2.full_name,
        "party_2": s2.party,
        "state_2": s2.state,
        "prompt_type": orientation,
        "comparison_prompt": cmp_prompt,
        "comparison_answer": cmp_answer,
        "extraction_prompt": ext_prompt,
        "extraction_answer": ext_answer,
        "extracted_name": extracted,
        "winner_conservative": winner,
    }

def build_win_matrix(results_df: pd.DataFrame, senators: List[Senator]) -> np.ndarray:
    n = len(senators)
    name_to_idx = {s.full_name: i for i, s in enumerate(senators)}
    W = np.zeros((n, n), dtype=float)
    for _, row in results_df.iterrows():
        i = name_to_idx[row["senator_1"]]
        j = name_to_idx[row["senator_2"]]
        winner = row["winner_conservative"]
        if winner == "Tie":
            W[i, j] += 0.5
            W[j, i] += 0.5
        elif winner == row["senator_1"]:
            W[i, j] += 1.0
        elif winner == row["senator_2"]:
            W[j, i] += 1.0
        else:
            W[i, j] += 0.5
            W[j, i] += 0.5
    return W

def fit_bradley_terry_mm(W: np.ndarray, max_iter: int = 5000, tol: float = 1e-10, epsilon: float = 1e-12):
    n = W.shape[0]
    N = W + W.T
    wins = W.sum(axis=1)
    p = np.ones(n, dtype=float)
    for _ in range(max_iter):
        p_old = p.copy()
        denom = np.zeros(n, dtype=float)
        for i in range(n):
            total = 0.0
            for j in range(n):
                if i == j or N[i, j] <= 0:
                    continue
                total += N[i, j] / (p[i] + p[j] + epsilon)
            denom[i] = total
        p = np.maximum(wins / np.maximum(denom, epsilon), epsilon)
        p /= math.exp(np.mean(np.log(p)))
        if np.max(np.abs(np.log(p + epsilon) - np.log(p_old + epsilon))) < tol:
            break
    lam = np.log(p)
    lam -= lam.mean()
    return p, lam

def rescale_unit_interval(values: np.ndarray) -> np.ndarray:
    vmin, vmax = float(values.min()), float(values.max())
    if math.isclose(vmin, vmax):
        return np.full_like(values, 0.5, dtype=float)
    return (values - vmin) / (vmax - vmin)

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    senators = load_senators(args.senators_csv)
    if args.limit is not None:
        senators = senators[:args.limit]
    print(f"Loaded {len(senators)} senators")

    judge = MistralJudge(
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    pairs = list(itertools.combinations(senators, 2))
    print(f"Total unique matchups: {len(pairs)}")

    all_rows = []
    for rep in range(args.repeats):
        print(f"\nStarting repeat {rep+1}/{args.repeats}")
        for k, (s1, s2) in enumerate(pairs, start=1):
            row = adjudicate_pair(judge, s1, s2)
            row["repeat"] = rep + 1
            all_rows.append(row)
            if k % args.save_every == 0 or k == len(pairs):
                pd.DataFrame(all_rows).to_csv(os.path.join(args.outdir, "pairwise_raw_results.csv"), index=False)
                print(f"  Saved progress: repeat {rep+1}, matchup {k}/{len(pairs)}")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    results_df = pd.DataFrame(all_rows)
    raw_path = os.path.join(args.outdir, "pairwise_raw_results.csv")
    results_df.to_csv(raw_path, index=False)

    W = build_win_matrix(results_df, senators)
    p, lam = fit_bradley_terry_mm(W)
    scaled = rescale_unit_interval(lam)

    scores = pd.DataFrame({
        "senator": [s.full_name for s in senators],
        "party": [s.party for s in senators],
        "state": [s.state for s in senators],
        "bt_alpha": p,
        "bt_lambda": lam,
        "lamp_score_unit_interval": scaled,
    }).sort_values("bt_lambda", ascending=False).reset_index(drop=True)
    scores.to_csv(os.path.join(args.outdir, "bradley_terry_scores.csv"), index=False)

    summary_rows = []
    grp = results_df.groupby(["senator_1", "senator_2"], sort=False)
    for (sen1, sen2), g in grp:
        wins_1 = wins_2 = 0.0
        for _, row in g.iterrows():
            if row["winner_conservative"] == "Tie":
                wins_1 += 0.5
                wins_2 += 0.5
            elif row["winner_conservative"] == sen1:
                wins_1 += 1.0
            elif row["winner_conservative"] == sen2:
                wins_2 += 1.0
            else:
                wins_1 += 0.5
                wins_2 += 0.5
        summary_rows.append({
            "senator_1": sen1,
            "senator_2": sen2,
            "wins_senator_1": wins_1,
            "wins_senator_2": wins_2,
            "n_repeats": len(g),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.outdir, "pairwise_matchup_summary.csv"), index=False)

    with open(os.path.join(args.outdir, "run_metadata.json"), "w") as f:
        json.dump({
            "senators_csv": args.senators_csv,
            "n_senators": len(senators),
            "n_unique_matchups": len(pairs),
            "repeats": args.repeats,
            "model_name_or_path": args.model_name_or_path,
            "greedy": args.greedy,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "note": "Prompt logic and tie coding follow Wu et al.; BT fitting here is classical MM, not BradleyTerry2 bias-reduced MLE.",
        }, f, indent=2)

    print("\nDone.")
    print(scores.head(10).to_string(index=False))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--senators-csv", type=str, default="data/116th_Senate_CFS.csv")
    ap.add_argument("--model-name-or-path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--outdir", type=str, default="mistral_bt_outputs")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--sleep-seconds", type=float, default=0.0)
    return ap.parse_args()

if __name__ == "__main__":
    run(parse_args())
