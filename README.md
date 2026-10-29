# SecureGenAI

**A Generative AI Framework for Vulnerability-Aware Code Generation and Secure Software Development**

Venkata Pruthvi Teja Pandugayala, Rohith Kumar Reddy Duggirala, Siddharth Arun
*IEEE Conference on Generative AI for Secure Systems (GAISS), 28–30 October 2026 · Paper ID 2809*

---

> **Status: reference implementation.** This repository implements the method described in the paper: data pipeline, model, joint loss, refinement loop and evaluation. It is **not** the code that produced the paper's published numbers, and it has not yet been trained on the full dataset to reproduce them. The full Vulnerability Fix Dataset is **not included**. `data/sample_pairs.csv` holds 20 hand-written illustrative pairs for smoke testing only. Results you get from this code will depend on your data and compute.

## Overview

AI code assistants learn from open-source code that contains vulnerable patterns, so they reproduce SQL injection, XSS, buffer overflows and similar flaws. Most existing defences detect flaws first and repair them afterwards. SecureGenAI builds vulnerability awareness into **generation**: it fine-tunes a transformer on real vulnerable → fixed code pairs annotated with CWE/CVE identifiers, and trains it with a joint generation + security objective.

```
vulnerable code ─► [1] input representation ─► [2] semantic encoder (multi-head attention)
                                                        │
                    ┌───────────────────────────────────┤
                    ▼                                   ▼
        [3] transformer decoder              [4] vulnerability refinement
            P(Y_s | X_v)                         S_vul = σ(W_h·H + b)
                    │                                   │
                    ▼                                   │
        [5] beam-search reconstruction ◄── feedback ────┘  (re-generate while S_vul > threshold)
                    │
                    ▼
               secure code
```

**Objective:** `L_total = λ1·L_gen + λ2·L_sec (+ λ_cwe·L_cwe)`, optimised with AdamW.

| Paper component | Where it lives |
|---|---|
| 6-step pre-processing (pair extraction → label masking) | `securegenai/data.py` |
| Vulnerability-aware input representation, semantic encoder, decoder | T5 backbone in `securegenai/model.py` |
| Vulnerability refinement module (`S_vul`, BCE security loss) | `SecureGenAI.security_head` |
| Secure code reconstruction (beam search, width 5) + feedback loop | `SecureGenAI.generate_secure` |
| Hyperparameters (Table I) | `configs/default.yaml` |
| Metrics (BLEU, detection rate, repair success, CWE accuracy/F1, ROC AUC) | `securegenai/metrics.py` |

## Repository layout

```
securegenai/
  config.py         configuration dataclass (Table I defaults)
  data.py           pre-processing pipeline, dataset, collator
  model.py          SecureGenAI model, joint loss, feedback-loop generation, save/load
  tokenization.py   T5 tokenizer (plus a tiny offline tokenizer for tests)
  metrics.py        evaluation metrics
  train.py          training CLI
  evaluate.py       evaluation CLI
  generate.py       fix a single snippet
configs/
  default.yaml      full T5-Small configuration
  smoke.yaml        tiny offline configuration (CPU, ~1 minute)
data/
  sample_pairs.csv  20 illustrative pairs (not the research dataset)
tests/              offline unit and end-to-end tests
```

## Installation

```bash
git clone https://github.com/<your-account>/SecureGenAI.git
cd SecureGenAI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is required. A CUDA GPU is recommended for full training. The paper used an NVIDIA T4 on Google Colab.

## Data format

Training expects a CSV with one vulnerable/fixed pair per row:

| column | required | example |
|---|---|---|
| `vulnerable_code` | yes | `cur.execute("SELECT * FROM users WHERE name = '" + name + "'")` |
| `fixed_code` | yes | `cur.execute("SELECT * FROM users WHERE name = %s", (name,))` |
| `cwe_id` | recommended | `CWE-89` |
| `language` | optional | `python` |
| `cve_id` | optional | `CVE-2021-44228` |

The paper's dataset has 35,000 pairs covering five classes: SQL injection (CWE-89), XSS (CWE-79), buffer overflow (CWE-120), OS command injection (CWE-78) and insecure cryptography (CWE-327). The languages are Java, Python, C and JavaScript. Column names and CWE classes are configurable in the YAML. Rows with an unknown CWE still train the generator and security head, and are skipped by the CWE classifier.

Pre-processing removes empty and duplicate pairs and pairs where the "fix" equals the vulnerable code. It then makes a 70 : 15 : 15 split (stratified by CWE where possible), tokenises with the T5 tokenizer (max 256 tokens), pads each batch dynamically, builds attention masks, and sets padded label positions to −100.

## Usage

**1. Smoke test (offline, CPU).** This checks the whole pipeline with a tiny random model. The numbers it prints are meaningless.

```bash
python -m pytest -q
python -m securegenai.train --config configs/smoke.yaml
```

**2. Train on your data**

```bash
python -m securegenai.train --config configs/default.yaml --set data_path=data/vulnerability_fix_dataset.csv
```

Outputs go to `outputs/run/`: the train/val/test splits, `history.csv`, `loss_curve.png`, and the best checkpoint (lowest validation loss) in `best/`. Any config value can be overridden with `--set key=value`.

**3. Evaluate on the held-out test split**

```bash
python -m securegenai.evaluate --checkpoint outputs/run/best --test outputs/run/test.csv
```

This writes `results.json` (all metrics, per-class scores, confusion matrix) and `predictions.csv` (every generated fix with its vulnerability score).

**4. Fix a snippet**

```bash
echo 'os.system("ping -c 1 " + host)' | python -m securegenai.generate --checkpoint outputs/run/best
```

**Google Colab:** in a GPU runtime, run `!git clone …`, `%cd SecureGenAI`, `!pip install -r requirements.txt`, then the train command above.

## Metrics

| Metric | Definition in this code |
|---|---|
| BLEU | corpus BLEU (sacrebleu) of generated vs reference fixes, on a 0–1 scale |
| Code-repair success rate | share of generated fixes that exactly match the reference after whitespace normalisation (strict) |
| Secure-output rate | share of generated fixes the refinement head scores at or below `vuln_threshold` |
| Vulnerability detection rate | share of held-out vulnerable snippets the refinement head flags (`S_vul > threshold`) |
| False-positive rate | share of held-out fixed snippets wrongly flagged |
| ROC AUC | refinement head, vulnerable vs fixed |
| Accuracy / precision / recall / F1 | CWE-class prediction (macro-averaged), with a 5 × 5 confusion matrix |

Exact match is a strict proxy for repair success: a correct fix written differently from the reference counts as a miss. Running generated code against compilers, unit tests or a SAST tool is a natural extension.

## Ablations

The paper's ablation (Table IV) removes components one at a time. These switches are available in config:

| Ablation | Setting |
|---|---|
| w/o security refinement | `use_security_head: false` (or `lambda_sec: 0`) |
| w/o feedback loop | `refine_iterations: 0` |
| w/o beam search | `beam_width: 1` (greedy) |
| two-term objective only | `use_cwe_head: false` |

## Implementation notes

These are the places where the paper leaves a detail open, and the choice made here:

- **Backbone.** T5-Small (d_model 512, 6 layers, 8 heads), as in Table I. T5 encodes position with relative position bias rather than added sinusoidal encodings.
- **Security loss.** The refinement head sees encoder representations of both the vulnerable input (label 1) and the fixed code (label 0), so it learns to separate the two. `H` is the masked mean of encoder states.
- **Loss weights.** The paper does not give values for λ1 and λ2. The defaults are λ1 = 1.0 and λ2 = 0.5.
- **CWE head.** An auxiliary CWE classifier (weight `lambda_cwe`) produces the per-class confusion matrix. Disable it for the plain two-term objective.
- **Feedback loop.** Generated code is re-scored by the refinement head. Outputs still above `vuln_threshold` are fed back in, up to `refine_iterations` extra passes.

## Results reported in the paper

These figures come from the paper and were **not** produced by this repository.

| Model | Accuracy | F1 | BLEU | Vuln. detection | Repair success |
|---|---|---|---|---|---|
| LSTM-based code repair | 84.26 | 82.79 | 0.71 | 81.63 | 80.94 |
| GRU-based secure generator | 89.34 | 88.11 | 0.79 | 87.63 | 86.58 |
| BERT code correction | 91.08 | 90.14 | 0.83 | 89.74 | 88.92 |
| CodeT5 baseline | 93.42 | 92.57 | 0.88 | 92.16 | 91.43 |
| **SecureGenAI** | **97.18** | **96.71** | **0.95** | **96.83** | **96.14** |

## Citation

```bibtex
@inproceedings{pandugayala2026securegenai,
  title     = {SecureGenAI: A Generative AI Framework for Vulnerability-Aware Code Generation and Secure Software Development},
  author    = {Pandugayala, Venkata Pruthvi Teja and Duggirala, Rohith Kumar Reddy and Arun, Siddharth},
  booktitle = {IEEE Conference on Generative AI for Secure Systems (GAISS)},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Responsible use

SecureGenAI suggests fixes. It does not guarantee that code is secure. Review every suggested patch, and keep static analysis, testing and code review in your workflow.
