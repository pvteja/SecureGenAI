"""Configuration for SecureGenAI.

Defaults follow Table I of the paper (T5-Small backbone, max length 256,
batch size 8, 30 epochs, AdamW at 5e-5 with weight decay 0.01, beam width 5).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CWE_CLASSES: List[str] = [
    "CWE-89",   # SQL injection
    "CWE-79",   # Cross-site scripting
    "CWE-120",  # Buffer overflow
    "CWE-78",   # OS command injection
    "CWE-327",  # Insecure / broken cryptography
]


@dataclass
class Config:
    # --- data -----------------------------------------------------------
    data_path: str = "data/sample_pairs.csv"
    vulnerable_col: str = "vulnerable_code"
    fixed_col: str = "fixed_code"
    cwe_col: str = "cwe_id"
    language_col: str = "language"
    cwe_classes: List[str] = field(default_factory=lambda: list(DEFAULT_CWE_CLASSES))
    split: List[float] = field(default_factory=lambda: [0.70, 0.15, 0.15])
    max_length: int = 256
    seed: int = 42

    # --- model ----------------------------------------------------------
    model_name: str = "t5-small"          # Hugging Face id or local path
    tiny_model: bool = False              # random tiny T5 for tests / offline smoke runs
    dropout: float = 0.1
    use_security_head: bool = True        # ablation: drop the vulnerability refinement module
    use_cwe_head: bool = True             # ablation: drop the auxiliary CWE classifier

    # --- loss -----------------------------------------------------------
    lambda_gen: float = 1.0               # λ1, generation (cross-entropy) loss
    lambda_sec: float = 0.5               # λ2, security (BCE) loss
    lambda_cwe: float = 0.5               # weight of auxiliary CWE classification loss

    # --- optimisation ---------------------------------------------------
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    device: str = "auto"

    # --- decoding -------------------------------------------------------
    beam_width: int = 5
    max_new_tokens: int = 256
    refine_iterations: int = 2            # feedback-loop passes (Fig. 1)
    vuln_threshold: float = 0.5

    # --- output ---------------------------------------------------------
    output_dir: str = "outputs/run"

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        raw.update(overrides or {})
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def parse_overrides(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """Turn ``["epochs=2", "tiny_model=true"]`` into a typed dict."""
    out: Dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = yaml.safe_load(value)
    return out
