"""Data pre-processing pipeline (Section III of the paper).

The six steps map onto this module as follows:

1. Pair extraction      -> :func:`load_pairs`
2. Sequence conversion  -> :func:`load_pairs` (every field is stringified)
3. Tokenise & truncate  -> :class:`VulnFixDataset` (T5 tokenizer, max 256 tokens)
4. Batch padding        -> :class:`PairCollator` (dynamic padding with the pad id)
5. Attention mask       -> :class:`PairCollator` (1 for tokens, 0 for padding)
6. Label masking        -> :class:`PairCollator` (pad positions in labels -> -100)

Expected CSV columns: ``vulnerable_code``, ``fixed_code``, ``cwe_id`` and
optionally ``language`` / ``cve_id``. Column names are configurable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import Config

IGNORE_INDEX = -100
UNKNOWN_CWE = -1


# ---------------------------------------------------------------- steps 1-2
def load_pairs(cfg: Config) -> pd.DataFrame:
    """Load the CSV, align vulnerable <-> fixed code and stringify every sample."""
    df = pd.read_csv(cfg.data_path)
    missing = {cfg.vulnerable_col, cfg.fixed_col} - set(df.columns)
    if missing:
        raise ValueError(f"{cfg.data_path} is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=[cfg.vulnerable_col, cfg.fixed_col]).copy()
    df[cfg.vulnerable_col] = df[cfg.vulnerable_col].astype(str).str.strip()
    df[cfg.fixed_col] = df[cfg.fixed_col].astype(str).str.strip()
    df = df[(df[cfg.vulnerable_col] != "") & (df[cfg.fixed_col] != "")]
    # A pair whose "fix" is identical to the vulnerable code teaches nothing.
    df = df[df[cfg.vulnerable_col] != df[cfg.fixed_col]]
    df = df.drop_duplicates(subset=[cfg.vulnerable_col, cfg.fixed_col])

    if cfg.cwe_col in df.columns:
        df[cfg.cwe_col] = df[cfg.cwe_col].astype(str).str.strip().str.upper()
    else:
        df[cfg.cwe_col] = "UNKNOWN"
    return df.reset_index(drop=True)


def split_pairs(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70:15:15 split, stratified by CWE class when every class is large enough."""
    train_frac, val_frac, test_frac = cfg.split
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(f"split must sum to 1, got {cfg.split}")

    def _strat(frame: pd.DataFrame):
        counts = frame[cfg.cwe_col].value_counts()
        return frame[cfg.cwe_col] if len(counts) > 1 and counts.min() >= 2 else None

    train, rest = train_test_split(
        df, train_size=train_frac, random_state=cfg.seed, stratify=_strat(df)
    )
    rel_val = val_frac / (val_frac + test_frac)
    try:
        val, test = train_test_split(
            rest, train_size=rel_val, random_state=cfg.seed, stratify=_strat(rest)
        )
    except ValueError:  # tiny datasets: fall back to an unstratified split
        val, test = train_test_split(rest, train_size=rel_val, random_state=cfg.seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ------------------------------------------------------------------ step 3
class VulnFixDataset(Dataset):
    """Tokenises each vulnerable/fixed pair, truncated to ``max_length`` tokens."""

    def __init__(self, df: pd.DataFrame, tokenizer, cfg: Config):
        self.cwe_to_id = {c.upper(): i for i, c in enumerate(cfg.cwe_classes)}
        vuln = df[cfg.vulnerable_col].tolist()
        fixed = df[cfg.fixed_col].tolist()
        self.cwe_ids = [self.cwe_to_id.get(c, UNKNOWN_CWE) for c in df[cfg.cwe_col].tolist()]
        self.vulnerable_text = vuln
        self.fixed_text = fixed

        enc_v = tokenizer(vuln, truncation=True, max_length=cfg.max_length)
        enc_f = tokenizer(fixed, truncation=True, max_length=cfg.max_length)
        self.vuln_ids: List[List[int]] = enc_v["input_ids"]
        self.fixed_ids: List[List[int]] = enc_f["input_ids"]

    def __len__(self) -> int:
        return len(self.vuln_ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "input_ids": self.vuln_ids[idx],
            "labels": self.fixed_ids[idx],
            "cwe_label": self.cwe_ids[idx],
        }


# -------------------------------------------------------------- steps 4-6
@dataclass
class PairCollator:
    """Dynamic padding, attention masks and -100 label masking.

    The fixed code is returned twice: as ``labels`` (decoder targets, pads set
    to -100) and as ``fixed_input_ids`` (encoder input with label 0 for the
    security loss, so the refinement head sees both vulnerable and secure code).
    """

    pad_token_id: int

    def _pad(self, seqs: Sequence[Sequence[int]], value: int) -> torch.Tensor:
        width = max(len(s) for s in seqs)
        out = torch.full((len(seqs), width), value, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        vuln = [b["input_ids"] for b in batch]
        fixed = [b["labels"] for b in batch]

        input_ids = self._pad(vuln, self.pad_token_id)
        fixed_input_ids = self._pad(fixed, self.pad_token_id)
        labels = fixed_input_ids.clone()
        labels[labels == self.pad_token_id] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_token_id).long(),
            "labels": labels,
            "fixed_input_ids": fixed_input_ids,
            "fixed_attention_mask": (fixed_input_ids != self.pad_token_id).long(),
            "cwe_label": torch.tensor([b["cwe_label"] for b in batch], dtype=torch.long),
        }


def make_loader(ds: VulnFixDataset, tokenizer, cfg: Config, shuffle: bool) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=PairCollator(tokenizer.pad_token_id),
        generator=gen if shuffle else None,
    )
