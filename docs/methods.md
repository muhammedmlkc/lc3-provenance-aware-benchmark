# Methods overview

## Validation schemes

| Label | Design | Intended interpretation |
|---|---|---|
| E0 | Repeated five-fold row-wise cross-validation | Prediction for another row from represented contexts |
| E1 | Repeated five-fold base-mixture-grouped cross-validation constrained within campaign | Prediction for an unseen mixture while the campaign remains represented in training |
| E3a | Leave-one-experimental-campaign-out cross-validation | Prediction for an unseen experimental campaign |
| E3b | Leave-one-research-programme-out cross-validation | Sensitivity to programme or laboratory grouping |

Campaigns containing only one base mixture are structurally ineligible for E1,
because their mixture cannot be held out while the same campaign remains in
training. They remain eligible for the other schemes.

The split generator projects each local input row onto identifiers only. It
rejects any attempted access to the response, reported dispersion, model
scores, or residuals. All assignments are deterministic functions of group
identifiers and fixed seed labels.

## Predictor panels

- **P0:** curing age, water-to-binder ratio, Portland cement or clinker share,
  calcined-clay share, limestone share, and harmonised Portland basis.
- **P1:** P0 plus gypsum, calcination temperature and duration,
  sand-to-binder ratio, and gypsum reporting status.
- **P2:** P1 plus curing temperatures, compression face area, curing classes,
  specimen shape, test-standard family, and aggregate class.

Identity variables, provenance identifiers, and target-derived variables are
never predictors.

## Learners and tuning

The configurations comprise mean, median, and log-age linear baselines; Ridge
with P0; and P0/P1/P2 versions of Elastic Net, Extra Trees, Histogram Gradient
Boosting, and CatBoost. Hyperparameters are selected in inner cross-validation
using group-macro mean absolute error. Imputation, scaling, and categorical
encoding are fitted only on the relevant training fold. Stochastic learners use
deterministic SHA-256-derived seeds and single-thread execution.

## Primary comparison

For each model-panel configuration, campaign-level E0 and E3a errors are paired
over their common campaigns. The central contrast is E3a minus E0
campaign-macro mean absolute error. The analysis code also produces a paired
campaign bootstrap interval and leave-one-campaign influence values.

These quantities describe the transferability of predictor information across
provenance boundaries. They do not identify causal material mechanisms.
