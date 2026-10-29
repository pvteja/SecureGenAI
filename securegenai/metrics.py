"""Evaluation metrics.

* BLEU                        - corpus BLEU (sacrebleu) of generated vs reference fixes, reported on a 0-1 scale
* Code-repair success rate    - share of generated fixes that match the reference after
                                whitespace normalisation (strict exact-match definition)
* Secure-output rate          - share of generated fixes the refinement head scores below threshold
* Vulnerability detection     - refinement head on held-out vulnerable vs fixed code:
                                detection rate (recall on vulnerable), false-positive rate, ROC AUC
* CWE classification          - accuracy, macro precision / recall / F1, confusion matrix
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def normalise_code(code: str) -> str:
    return re.sub(r"\s+", " ", code).strip()


def bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    import sacrebleu

    if not predictions:
        return 0.0
    score = sacrebleu.corpus_bleu(list(predictions), [list(references)], tokenize="13a")
    return score.score / 100.0


def repair_success_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    hits = sum(normalise_code(p) == normalise_code(r) for p, r in zip(predictions, references))
    return hits / len(predictions)


def detection_metrics(vuln_scores: Sequence[float], fixed_scores: Sequence[float], threshold: float) -> Dict[str, float]:
    v = np.asarray(vuln_scores, dtype=float)
    f = np.asarray(fixed_scores, dtype=float)
    y_true = np.concatenate([np.ones_like(v), np.zeros_like(f)])
    y_score = np.concatenate([v, f])
    out = {
        "vulnerability_detection_rate": float((v > threshold).mean()) if len(v) else 0.0,
        "false_positive_rate_on_fixed": float((f > threshold).mean()) if len(f) else 0.0,
        "detection_accuracy": float(((y_score > threshold) == y_true).mean()) if len(y_true) else 0.0,
    }
    out["roc_auc"] = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) == 2 else float("nan")
    return out


def cwe_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: List[str]) -> Dict[str, object]:
    labels = list(range(len(class_names)))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "per_class": {
            name: {"precision": float(a), "recall": float(b), "f1": float(c), "support": int(s)}
            for name, a, b, c, s in zip(class_names, per_p, per_r, per_f1, support)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": class_names,
    }
