# Provenance-aware validation benchmark for LC3 mortar

This repository contains the computational methods for the study:

> **Random row-wise cross-validation underestimates prediction error for unseen
> experimental campaigns in LC3 mortar: a provenance-aware benchmark**

The workflow compares row-wise, mixture-grouped, campaign-held-out, and
programme-group-held-out validation. Its purpose is to align validation groups
with the intended prediction setting; it does not introduce a new regression
algorithm.

## Public release boundary

This is deliberately a code-only release. It contains author-created software,
method documentation, a frozen model-candidate grid, an environment record, and
tests that generate artificial data at runtime.

It does **not** contain:

- source PDFs, screenshots, or extracted tables;
- source-derived observations or provenance rows;
- true targets, model predictions, split manifests, or fitted models;
- aggregate results or publication figures; or
- hashes, locators, or counts that identify the private research corpus.

The dataset used in the accompanying study was reconstructed from third-party
publications with heterogeneous reuse terms. It is therefore not sublicensed or
redistributed here. Users who wish to reproduce numerical results must lawfully
reconstruct an eligible dataset from the cited primary sources.

## Contents

| Path | Contents |
|---|---|
| `scripts/` | Target-blind split generation, nested validation, aggregation, and release validation |
| `tests/` | Implementation tests based entirely on artificial records |
| `config/model_candidate_grid.json` | Frozen hyperparameter candidates |
| `config/environment.json` | Recorded software environment and thread policy |
| `docs/` | Input schema, method summary, and local-use instructions |
| `RELEASE_MANIFEST.sha256` | Hashes of every distributed file |

## Environment

Python 3.11 and Node.js are required. The exact Python packages recorded for
the study are listed in `requirements-lock.txt`.

```bash
python -m venv .venv
python -m pip install -r requirements-lock.txt
python scripts/validate_release.py
python tests/test_pipeline.py
```

The release validator checks the code-only boundary, manifest integrity,
private paths, and common secret formats. The implementation tests create all
identifiers and numeric values in a temporary directory and delete them when
the test process finishes.

## Running the workflow on a lawfully prepared local dataset

Prepare a model-ready CSV and a hierarchy CSV using the fields in
`docs/data_dictionary.md`. Keep both outside the repository or in an ignored
local directory.

```bash
node scripts/generate_splits.mjs \
  --input /path/to/model_ready.csv \
  --programme-map /path/to/hierarchy.csv \
  --out /path/to/private_splits

python scripts/run_benchmark.py \
  --validate-only \
  --input /path/to/model_ready.csv \
  --split-manifest /path/to/private_splits/split_manifest.csv \
  --repeated-manifest /path/to/private_splits/split_manifest_repeated.csv
```

Model fitting is deliberately protected by an explicit flag:

```bash
python scripts/run_benchmark.py \
  --execute \
  --input /path/to/model_ready.csv \
  --split-manifest /path/to/private_splits/split_manifest.csv \
  --repeated-manifest /path/to/private_splits/split_manifest_repeated.csv \
  --output-dir /path/to/private_outputs

python scripts/analyze_results.py \
  --results-dir /path/to/private_outputs \
  --output-dir /path/to/private_analysis
```

Generated manifests, predictions, summaries, figures, and reconstructed data
must remain outside this public repository unless the depositor independently
holds the necessary redistribution rights.

## Citation and licence

Citation metadata are provided in `CITATION.cff`. The MIT License applies only
to the original software and documentation in this repository. It grants no
rights over third-party publications or data reconstructed from them.
