"""Evaluate a trained SecureGenAI checkpoint on the held-out test split.

    python -m securegenai.evaluate --checkpoint outputs/run/best --test outputs/run/test.csv

Writes ``results.json`` and ``predictions.csv`` next to the checkpoint (or to
``--out``).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import torch

from .data import UNKNOWN_CWE, VulnFixDataset, make_loader
from .metrics import bleu, cwe_metrics, detection_metrics, repair_success_rate
from .model import load_model
from .train import resolve_device


@torch.no_grad()
def evaluate(checkpoint: str, test_csv: str, out_dir: str | None = None,
             refine_iterations: int | None = None, limit: int | None = None) -> Dict[str, object]:
    device = resolve_device("auto")
    model, tokenizer, cfg = load_model(checkpoint, device=str(device))
    df = pd.read_csv(test_csv)
    if limit:
        df = df.head(limit)
    ds = VulnFixDataset(df, tokenizer, cfg)
    loader = make_loader(ds, tokenizer, cfg, shuffle=False)

    vuln_scores: List[float] = []
    fixed_scores: List[float] = []
    cwe_true: List[int] = []
    cwe_pred: List[int] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if model.security_head is not None:
            vuln_scores += model.vulnerability_score(batch["input_ids"], batch["attention_mask"]).tolist()
            fixed_scores += model.vulnerability_score(batch["fixed_input_ids"], batch["fixed_attention_mask"]).tolist()
        if model.cwe_head is not None:
            pred = model.predict_cwe(batch["input_ids"], batch["attention_mask"]).argmax(-1)
            keep = batch["cwe_label"] != UNKNOWN_CWE
            cwe_true += batch["cwe_label"][keep].tolist()
            cwe_pred += pred[keep].tolist()

    # Generation with beam search + feedback loop, in batches.
    generations: List[Dict[str, object]] = []
    texts = ds.vulnerable_text
    for start in range(0, len(texts), cfg.batch_size):
        generations += model.generate_secure(tokenizer, texts[start:start + cfg.batch_size],
                                             refine_iterations=refine_iterations)
    preds = [g["output"] for g in generations]
    refs = ds.fixed_text

    results: Dict[str, object] = {
        "n_test": len(df),
        "bleu": bleu(preds, refs),
        "code_repair_success_rate": repair_success_rate(preds, refs),
    }
    gen_scores = [g["vuln_score"] for g in generations if g["vuln_score"] is not None]
    if gen_scores:
        results["secure_output_rate"] = sum(s <= cfg.vuln_threshold for s in gen_scores) / len(gen_scores)
    if vuln_scores:
        results.update(detection_metrics(vuln_scores, fixed_scores, cfg.vuln_threshold))
    if cwe_true:
        results["cwe_classification"] = cwe_metrics(cwe_true, cwe_pred, cfg.cwe_classes)

    out_dir = out_dir or checkpoint
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    pd.DataFrame({
        "vulnerable_code": texts,
        "reference_fix": refs,
        "generated_fix": preds,
        "vuln_score_after": [g["vuln_score"] for g in generations],
        "passes": [g["passes"] for g in generations],
    }).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate SecureGenAI")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test", required=True, help="test split CSV written by train.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--refine-iterations", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    args = ap.parse_args()
    res = evaluate(args.checkpoint, args.test, args.out, args.refine_iterations, args.limit)
    summary = {k: v for k, v in res.items() if k != "cwe_classification"}
    if "cwe_classification" in res:
        c = res["cwe_classification"]
        summary.update({f"cwe_{k}": c[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
