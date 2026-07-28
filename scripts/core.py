"""Analysis components for the LC3 validation benchmark.

Importing this module does not fit a model. All preprocessing is constructed
inside a training fold, and split construction is explicitly target-blind.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "compressive_strength_mpa"
FORBIDDEN_SPLIT_FIELDS = frozenset({
    TARGET,
    "compressive_strength_sd_mpa",
    "reported_dispersion_mpa",
    "reported_dispersion_statistic",
    "model_score",
    "residual",
})
P0_NUMERIC = (
    "curing_age_days",
    "water_to_binder_ratio",
    "portland_or_clinker_pct_binder",
    "calcined_clay_pct_binder",
    "limestone_pct_binder",
)
P0_CATEGORICAL = ("portland_basis_harmonized",)
P1_NUMERIC = P0_NUMERIC + (
    "gypsum_pct_binder",
    "calcination_temperature_c",
    "calcination_duration_h",
    "sand_to_binder_ratio",
)
P1_CATEGORICAL = P0_CATEGORICAL + ("gypsum_reporting_status",)
P2_NUMERIC = P1_NUMERIC + (
    "initial_cure_temperature_c",
    "post_demould_temperature_c",
    "compression_face_area_mm2_exact",
)
P2_CATEGORICAL = P1_CATEGORICAL + (
    "initial_cure_class",
    "post_demould_cure_class",
    "specimen_shape_class",
    "test_standard_family",
    "aggregate_class",
)


class TargetAccessError(RuntimeError):
    """Raised when split construction attempts to read an outcome field."""


class RunBlocked(RuntimeError):
    """Raised when a frozen prerequisite or hash binding is not satisfied."""


class CatBoostSklearnAdapter(RegressorMixin, BaseEstimator):
    """Expose CatBoost through the estimator protocol required by sklearn 1.8."""

    def __init__(self, parameters: Mapping[str, object], seed: int):
        self.parameters = parameters
        self.seed = seed

    def fit(self, features, target):
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            random_seed=self.seed,
            **dict(self.parameters),
        )
        self.model_.fit(features, target)
        self.n_features_in_ = np.asarray(features).shape[1]
        return self

    def predict(self, features):
        if not hasattr(self, "model_"):
            raise RuntimeError("CatBoostSklearnAdapter is not fitted")
        return np.asarray(self.model_.predict(features), dtype=float)


class TargetBlindMapping(Mapping[str, object]):
    """Read-only mapping that fails loudly on any outcome-like field access."""

    def __init__(self, row: Mapping[str, object]):
        self._row = row

    def __getitem__(self, key: str) -> object:
        if key in FORBIDDEN_SPLIT_FIELDS:
            raise TargetAccessError(f"TARGET_ACCESS_PROHIBITED: {key}")
        return self._row[key]

    def __iter__(self):
        return (key for key in self._row if key not in FORBIDDEN_SPLIT_FIELDS)

    def __len__(self) -> int:
        return sum(key not in FORBIDDEN_SPLIT_FIELDS for key in self._row)


def project_split_fields(row: Mapping[str, object]) -> dict[str, object]:
    """Project a source row onto target-independent hierarchy fields."""
    protected = TargetBlindMapping(row)
    return {
        key: protected[key]
        for key in (
            "record_id", "base_mix_id", "publication_family_id",
            "experimental_campaign_id", "research_programme_id",
        )
    }


def make_fold_local_preprocessor(
    numeric_fields: Sequence[str], categorical_fields: Sequence[str]
) -> ColumnTransformer:
    """Return an unfitted preprocessor whose state can only come from ``fit`` data."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("numeric", numeric, list(numeric_fields)), ("categorical", categorical, list(categorical_fields))],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def fitted_preprocessor_state(preprocessor: ColumnTransformer) -> dict[str, object]:
    """Expose the learned fold-local state for leakage audit tests."""
    numeric = preprocessor.named_transformers_["numeric"]
    categorical = preprocessor.named_transformers_["categorical"]
    imputer = numeric.named_steps["imputer"]
    scaler = numeric.named_steps["scaler"]
    encoder = categorical.named_steps["encoder"]
    return {
        "numeric_imputer_statistics": np.asarray(imputer.statistics_).tolist(),
        "numeric_scaler_mean": np.asarray(scaler.mean_).tolist(),
        "categorical_levels": [np.asarray(levels).astype(str).tolist() for levels in encoder.categories_],
    }


def panel_fields(panel: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    panels = {
        "P0": (P0_NUMERIC, P0_CATEGORICAL),
        "P1": (P1_NUMERIC, P1_CATEGORICAL),
        "P2": (P2_NUMERIC, P2_CATEGORICAL),
    }
    try:
        return panels[panel]
    except KeyError as exc:
        raise ValueError(f"Unknown panel {panel!r}; expected P0, P1 or P2") from exc


def seed_from_labels(*labels: object) -> int:
    digest = hashlib.sha256("::".join(map(str, labels)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_hash_bindings(paths: Iterable[str | Path]) -> dict[str, str]:
    return {Path(path).name: sha256_file(path) for path in paths}


def verify_hash_bindings(expected: Mapping[str, str], roots: Iterable[str | Path]) -> None:
    actual = build_hash_bindings(roots)
    if dict(expected) != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(name for name in expected.keys() & actual.keys() if expected[name].upper() != actual[name].upper())
        raise RunBlocked(f"HASH_LOCK_MISMATCH missing={missing} extra={extra} changed={changed}")


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    error = truth - pred
    absolute = np.abs(error)
    result = {
        "mae": float(np.mean(absolute)),
        "medae": float(np.median(absolute)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    result["r2"] = float(1 - np.sum(error ** 2) / denominator) if len(truth) >= 5 and denominator > 0 else math.nan
    return result


def group_error_table(
    frame: pd.DataFrame,
    group_field: str,
    target_field: str = "y_true",
    prediction_field: str = "y_pred",
) -> pd.DataFrame:
    rows = []
    for group, subset in frame.groupby(group_field, sort=True, dropna=False):
        metrics = regression_metrics(subset[target_field], subset[prediction_field])
        rows.append({group_field: group, "n": len(subset), **metrics})
    return pd.DataFrame(rows)


def macro_error_summary(group_table: pd.DataFrame) -> dict[str, float]:
    mae = group_table["mae"].to_numpy(dtype=float)
    return {
        "macro_mae": float(np.mean(mae)),
        "worst_group_mae": float(np.max(mae)),
        "upper_quartile_group_mae": float(np.quantile(mae, 0.75)),
        "group_count": int(len(mae)),
    }


def campaign_cluster_bootstrap_contrast(
    paired_errors: pd.DataFrame,
    e0_field: str,
    e3a_field: str,
    draws: int = 10_000,
    seed_label: str = "CAMPAIGN_BOOTSTRAP_V1",
) -> dict[str, object]:
    """Bootstrap a paired campaign-level E3a minus E0 MAE contrast."""
    if draws < 10_000:
        raise ValueError("The frozen protocol requires at least 10,000 draws")
    differences = (paired_errors[e3a_field] - paired_errors[e0_field]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed_from_labels(seed_label))
    sampled = rng.choice(differences, size=(draws, len(differences)), replace=True).mean(axis=1)
    return {
        "observed_mean_difference": float(differences.mean()),
        "percentile_95_interval": np.quantile(sampled, [0.025, 0.975]).astype(float).tolist(),
        "draws": draws,
        "seed_label": seed_label,
    }


def leave_one_campaign_influence(paired_errors: pd.DataFrame, e0_field: str, e3a_field: str) -> pd.DataFrame:
    differences = (paired_errors[e3a_field] - paired_errors[e0_field]).to_numpy(dtype=float)
    result = []
    for position, index in enumerate(paired_errors.index):
        retained = np.delete(differences, position)
        result.append({"held_out_index": index, "contrast_without_campaign": float(retained.mean())})
    return pd.DataFrame(result)


def gower_min_distance_p0(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_fields: Sequence[str] = P0_NUMERIC,
    categorical_fields: Sequence[str] = P0_CATEGORICAL,
) -> np.ndarray:
    """Compute fold-local minimum Gower distance using training ranges only."""
    numeric_train = train[list(numeric_fields)].astype(float)
    numeric_test = test[list(numeric_fields)].astype(float)
    medians = numeric_train.median(axis=0)
    numeric_train = numeric_train.fillna(medians)
    numeric_test = numeric_test.fillna(medians)
    ranges = numeric_train.max(axis=0) - numeric_train.min(axis=0)
    output = []
    for test_index in range(len(test)):
        components = []
        for field in numeric_fields:
            delta = np.abs(numeric_train[field].to_numpy() - numeric_test.iloc[test_index][field])
            span = float(ranges[field])
            if span > 0:
                components.append(delta / span)
            else:
                components.append((delta != 0).astype(float))
        for field in categorical_fields:
            train_values = train[field].fillna("__MISSING__").astype(str).to_numpy()
            test_value = str(test.iloc[test_index][field]) if pd.notna(test.iloc[test_index][field]) else "__MISSING__"
            components.append((train_values != test_value).astype(float))
        distance = np.vstack(components).mean(axis=0)
        output.append(float(distance.min()))
    return np.asarray(output)


@dataclass(frozen=True)
class Candidate:
    model: str
    parameters: Mapping[str, object]

    @property
    def ordering_hash(self) -> str:
        canonical = json.dumps(dict(self.parameters), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{self.model}::{canonical}".encode("utf-8")).hexdigest().upper()


def ordered_candidates(model: str, parameter_rows: Sequence[Mapping[str, object]]) -> list[Candidate]:
    return sorted((Candidate(model, dict(parameters)) for parameters in parameter_rows), key=lambda item: item.ordering_hash)


def make_regressor(model: str, parameters: Mapping[str, object], seed: int):
    """Instantiate one frozen model candidate in single-thread CPU mode."""
    values = dict(parameters)
    if model == "ridge":
        return Ridge(**values)
    if model == "elastic_net":
        return ElasticNet(max_iter=100_000, selection="cyclic", random_state=seed, **values)
    if model == "extra_trees":
        return ExtraTreesRegressor(n_estimators=500, n_jobs=1, random_state=seed, **values)
    if model == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed, max_iter=300, early_stopping=False, **values)
    if model == "catboost":
        return CatBoostSklearnAdapter(parameters=values, seed=seed)
    raise ValueError(f"Unknown model {model!r}")


def make_model_pipeline(
    model: str,
    parameters: Mapping[str, object],
    seed: int,
    panel: str,
) -> Pipeline:
    numeric, categorical = panel_fields(panel)
    return Pipeline([
        ("preprocess", make_fold_local_preprocessor(numeric, categorical)),
        ("regressor", make_regressor(model, parameters, seed)),
    ])


def assert_group_integrity(frame: pd.DataFrame, group_field: str, fold_field: str) -> None:
    crossing = frame.groupby(group_field, dropna=False)[fold_field].nunique()
    offenders = crossing[crossing != 1]
    if not offenders.empty:
        raise AssertionError(f"{group_field} crosses {fold_field}: {offenders.index.tolist()}")
