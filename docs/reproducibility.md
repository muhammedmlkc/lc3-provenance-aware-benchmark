# Reproducibility and local execution

## Code-only release

The public package contains the executable method but no populated research
data, provenance table, split manifest, prediction, fitted model, aggregate
result, or publication figure. This boundary prevents the repository licence
from being mistaken for permission to redistribute material reconstructed from
third-party publications.

## Deterministic split construction

`scripts/generate_splits.mjs` reads a model-ready CSV and a separate hierarchy
CSV supplied by the user. The generator exposes only record, mixture,
publication, campaign, and programme identifiers to its assignment functions.
Access to outcome-like fields raises an error. The generated files include
single and repeated split manifests, a group-assignment audit, a validation
summary, and their hashes; all are written to a user-selected local directory.

## Fold-local preprocessing and tuning

`scripts/run_benchmark.py` checks the input schema and agreement between the
model-ready input and both split manifests. Model fitting occurs only when
`--execute` is supplied. Every imputer, scaler, encoder, and learner is fitted
within its training fold. Candidate selection uses only inner-fold scores.

The model grid is frozen in `config/model_candidate_grid.json`. The recorded
software and threading policy is in `config/environment.json`.

## Artificial implementation tests

`tests/test_pipeline.py` generates an artificial hierarchy and response in an
operating-system temporary directory. It verifies group integrity,
target-permutation invariance, target-access rejection, fold-local
preprocessing, deterministic generation, hash-change rejection, and a known
row-wise versus campaign-held-out validation gap. The tests print a report to
standard output and do not create tracked artifacts.

## Result aggregation

`scripts/analyze_results.py` accepts a private benchmark-output directory and
writes summaries to another user-selected directory. It computes pooled and
group-macro metrics, paired E3a-minus-E0 contrasts, deterministic paired
bootstrap intervals, leave-one-campaign influence, E1-versus-E0 matched
comparisons, model-versus-age comparisons, panel sensitivity, seed variation,
and hyperparameter-selection frequencies.

Generated files should not be committed to this repository unless the person
publishing them has independently verified the applicable rights and disclosure
requirements.
