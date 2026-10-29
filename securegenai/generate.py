"""Suggest a secure version of a code snippet.

    python -m securegenai.generate --checkpoint outputs/run/best --file snippet.py
    echo 'query = "SELECT * FROM users WHERE id=" + uid' | python -m securegenai.generate --checkpoint outputs/run/best
"""
from __future__ import annotations

import argparse
import sys

from .model import load_model
from .train import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a secure fix with SecureGenAI")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--file", default=None, help="file with the vulnerable snippet (default: stdin)")
    ap.add_argument("--beam-width", type=int, default=None)
    ap.add_argument("--refine-iterations", type=int, default=None)
    args = ap.parse_args()

    code = open(args.file, encoding="utf-8").read() if args.file else sys.stdin.read()
    model, tokenizer, cfg = load_model(args.checkpoint, device=str(resolve_device("auto")))

    enc = tokenizer([code], truncation=True, max_length=cfg.max_length, return_tensors="pt").to(
        next(model.parameters()).device)
    if model.security_head is not None:
        before = float(model.vulnerability_score(enc["input_ids"], enc["attention_mask"])[0])
        print(f"# vulnerability score (input): {before:.3f}")
    if model.cwe_head is not None:
        logits = model.predict_cwe(enc["input_ids"], enc["attention_mask"])[0]
        print(f"# predicted CWE class: {cfg.cwe_classes[int(logits.argmax())]}")

    res = model.generate_secure(tokenizer, [code], beam_width=args.beam_width,
                                refine_iterations=args.refine_iterations)[0]
    if res["vuln_score"] is not None:
        print(f"# vulnerability score (output): {res['vuln_score']:.3f} after {res['passes']} pass(es)")
    print(res["output"])


if __name__ == "__main__":
    main()
