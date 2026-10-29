# Data

`sample_pairs.csv` has 20 hand-written vulnerable/fixed pairs: four for each of the five CWE classes, across Python, Java, C and JavaScript. They exist so the pipeline and tests run out of the box. They are **not** the research dataset and are far too few to train a useful model.

To train for real, put the full Vulnerability Fix Dataset CSV here (for example `data/vulnerability_fix_dataset.csv`) and pass `--set data_path=...`. Other CSVs in this folder are git-ignored so research data isn't committed by accident.

Required columns: `vulnerable_code`, `fixed_code`. Recommended: `cwe_id`. Optional: `language`, `cve_id`.
