"""Train SecureGenAI.

    python -m securegenai.train --config configs/default.yaml
    python -m securegenai.train --config configs/default.yaml --set epochs=5 batch_size=16

Writes to ``output_dir``: the train/val/test splits, ``history.csv`` (per-epoch
losses), ``loss_curve.png`` and the best checkpoint (lowest validation loss)
under ``best/``.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from typing import Dict

import numpy as np
import torch

from .config import Config, parse_overrides
from .data import VulnFixDataset, load_pairs, make_loader, split_pairs
from .model import SecureGenAI, build_model, save_model
from .tokenization import build_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(model: SecureGenAI, loader, device, optimizer=None, grad_clip: float = 1.0) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "gen_loss": 0.0, "sec_loss": 0.0, "cwe_loss": 0.0}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out = model(**batch)
            if train:
                optimizer.zero_grad(set_to_none=True)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        bs = batch["input_ids"].size(0)
        n += bs
        for key in totals:
            totals[key] += float(getattr(out, key).detach()) * bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def plot_history(path_csv: str, path_png: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs, tr, va = [], [], []
    with open(path_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row["epoch"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, tr, label="Training loss")
    ax.plot(epochs, va, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("SecureGenAI training convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def train(cfg: Config) -> str:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.output_dir, "config.yaml"))

    df = load_pairs(cfg)
    train_df, val_df, test_df = split_pairs(df, cfg)
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        part.to_csv(os.path.join(cfg.output_dir, f"{name}.csv"), index=False)
    print(f"Loaded {len(df)} pairs -> train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")

    tokenizer = build_tokenizer(cfg)
    train_loader = make_loader(VulnFixDataset(train_df, tokenizer, cfg), tokenizer, cfg, shuffle=True)
    val_loader = make_loader(VulnFixDataset(val_df, tokenizer, cfg), tokenizer, cfg, shuffle=False)

    model = build_model(cfg, tokenizer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    history_path = os.path.join(cfg.output_dir, "history.csv")
    best_dir = os.path.join(cfg.output_dir, "best")
    best_val = math.inf
    with open(history_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_gen_loss", "val_sec_loss", "val_cwe_loss", "seconds"])
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, train_loader, device, optimizer, cfg.grad_clip)
            va = run_epoch(model, val_loader, device)
            secs = time.time() - t0
            writer.writerow([epoch, f"{tr['loss']:.6f}", f"{va['loss']:.6f}", f"{va['gen_loss']:.6f}",
                             f"{va['sec_loss']:.6f}", f"{va['cwe_loss']:.6f}", f"{secs:.1f}"])
            fh.flush()
            marker = ""
            if va["loss"] < best_val:
                best_val = va["loss"]
                save_model(model, tokenizer, best_dir)
                marker = "  <- best"
            print(f"epoch {epoch:3d} | train {tr['loss']:.4f} | val {va['loss']:.4f} "
                  f"(gen {va['gen_loss']:.4f}, sec {va['sec_loss']:.4f}, cwe {va['cwe_loss']:.4f}) | {secs:.0f}s{marker}")

    plot_history(history_path, os.path.join(cfg.output_dir, "loss_curve.png"))
    print(f"Best validation loss {best_val:.4f}; checkpoint saved to {best_dir}")
    return best_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SecureGenAI")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    args = ap.parse_args()
    train(Config.from_yaml(args.config, parse_overrides(args.set)))


if __name__ == "__main__":
    main()
