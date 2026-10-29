"""Offline tests: tiny random T5 + character tokenizer, no downloads needed."""
from __future__ import annotations

import json
import math
import os

import pandas as pd
import pytest
import torch

from securegenai.config import Config
from securegenai.data import IGNORE_INDEX, PairCollator, VulnFixDataset, load_pairs, make_loader, split_pairs
from securegenai.metrics import bleu, detection_metrics, repair_success_rate
from securegenai.model import build_model, load_model, save_model
from securegenai.tokenization import build_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def cfg(tmp_path) -> Config:
    return Config(
        data_path=os.path.join(ROOT, "data", "sample_pairs.csv"),
        tiny_model=True, max_length=96, batch_size=4, epochs=1,
        learning_rate=1e-3, beam_width=2, max_new_tokens=16, refine_iterations=1,
        output_dir=str(tmp_path / "run"),
    )


def test_load_and_split(cfg):
    df = load_pairs(cfg)
    assert len(df) == 20
    tr, va, te = split_pairs(df, cfg)
    assert len(tr) + len(va) + len(te) == len(df)
    assert len(tr) == 14
    assert set(tr["vulnerable_code"]).isdisjoint(set(te["vulnerable_code"]))


def test_load_drops_bad_rows(cfg, tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({
        "vulnerable_code": ["a", "same", None, "x", "x"],
        "fixed_code": ["b", "same", "c", "y", "y"],
        "cwe_id": ["cwe-89"] * 5,
    }).to_csv(p, index=False)
    cfg.data_path = str(p)
    df = load_pairs(cfg)
    # identical pair, missing value and duplicate are dropped; CWE is upper-cased
    assert df["vulnerable_code"].tolist() == ["a", "x"]
    assert df["cwe_id"].unique().tolist() == ["CWE-89"]


def test_collator_masks(cfg):
    tok = build_tokenizer(cfg)
    df = load_pairs(cfg).head(4)
    ds = VulnFixDataset(df, tok, cfg)
    batch = PairCollator(tok.pad_token_id)([ds[i] for i in range(4)])
    pad = tok.pad_token_id
    assert torch.equal(batch["attention_mask"], (batch["input_ids"] != pad).long())
    # labels: pad positions -> -100, real tokens unchanged
    real = batch["fixed_input_ids"] != pad
    assert torch.all(batch["labels"][~real] == IGNORE_INDEX)
    assert torch.equal(batch["labels"][real], batch["fixed_input_ids"][real])
    assert batch["input_ids"].shape[1] <= cfg.max_length


def test_tokenizer_roundtrip(cfg):
    tok = build_tokenizer(cfg)
    code = 'cur.execute("SELECT * FROM t WHERE id = %s", (uid,))'
    ids = tok(code)["input_ids"]
    assert ids[-1] == tok.eos_token_id
    assert tok.decode(ids, skip_special_tokens=True) == code


def test_forward_losses_and_training_step(cfg):
    torch.manual_seed(0)
    tok = build_tokenizer(cfg)
    df = load_pairs(cfg)
    loader = make_loader(VulnFixDataset(df, tok, cfg), tok, cfg, shuffle=False)
    model = build_model(cfg, tok)
    batch = next(iter(loader))
    out = model(**batch)
    for name in ("loss", "gen_loss", "sec_loss", "cwe_loss"):
        assert math.isfinite(float(getattr(out, name))), name
    assert out.sec_loss > 0 and out.cwe_loss > 0

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = float(out.loss)
    for _ in range(15):
        opt.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        opt.step()
    assert float(model(**batch).loss) < first


def test_ablation_without_heads(cfg):
    cfg.use_security_head = False
    cfg.use_cwe_head = False
    tok = build_tokenizer(cfg)
    loader = make_loader(VulnFixDataset(load_pairs(cfg), tok, cfg), tok, cfg, shuffle=False)
    model = build_model(cfg, tok)
    out = model(**next(iter(loader)))
    assert float(out.sec_loss) == 0.0 and float(out.cwe_loss) == 0.0
    res = model.generate_secure(tok, ["os.system('ls ' + d)"])
    assert res[0]["vuln_score"] is None and res[0]["passes"] == 1


def test_generate_feedback_loop(cfg):
    tok = build_tokenizer(cfg)
    model = build_model(cfg, tok).eval()
    # threshold -1 forces every output back through the loop
    res = model.generate_secure(tok, ["gets(name);", "strcpy(b, s);"], threshold=-1.0, refine_iterations=2)
    assert all(r["passes"] == 3 for r in res)
    assert all(0.0 <= r["vuln_score"] <= 1.0 for r in res)
    # threshold 2 means one pass only
    res = model.generate_secure(tok, ["gets(name);"], threshold=2.0, refine_iterations=2)
    assert res[0]["passes"] == 1


def test_save_load_roundtrip(cfg, tmp_path):
    tok = build_tokenizer(cfg)
    model = build_model(cfg, tok).eval()
    path = str(tmp_path / "ckpt")
    save_model(model, tok, path)
    loaded, tok2, cfg2 = load_model(path)
    enc = tok(["eval(x)"], return_tensors="pt")
    a = model.vulnerability_score(enc["input_ids"], enc["attention_mask"])
    b = loaded.vulnerability_score(enc["input_ids"], enc["attention_mask"])
    assert torch.allclose(a, b)
    assert cfg2.cwe_classes == cfg.cwe_classes


def test_metrics():
    assert repair_success_rate(["a  b", "c"], ["a b", "d"]) == 0.5
    s = "result = compute(alpha, beta, gamma) + offset"
    assert bleu([s], [s]) == pytest.approx(1.0)
    m = detection_metrics([0.9, 0.8], [0.1, 0.6], threshold=0.5)
    assert m["vulnerability_detection_rate"] == 1.0
    assert m["false_positive_rate_on_fixed"] == 0.5
    assert m["roc_auc"] == 1.0


def test_train_and_evaluate_end_to_end(cfg):
    from securegenai.evaluate import evaluate
    from securegenai.train import train

    best = train(cfg)
    assert os.path.exists(os.path.join(cfg.output_dir, "history.csv"))
    res = evaluate(best, os.path.join(cfg.output_dir, "test.csv"))
    for key in ("bleu", "code_repair_success_rate", "vulnerability_detection_rate", "roc_auc", "cwe_classification"):
        assert key in res
    with open(os.path.join(best, "results.json")) as fh:
        assert json.load(fh)["n_test"] == res["n_test"]
