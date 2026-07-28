"""Executable nested-validation runner for the LC3 benchmark.

The module is inert when imported. Model fitting requires the explicit
``--execute`` flag and user-supplied local input and split manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression

import core


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = REPO_ROOT / "config" / "model_candidate_grid.json"
STOCHASTIC_MODELS = frozenset({"extra_trees", "catboost"})
MODEL_SEED_LABELS = tuple(f"MODEL_SEED_R{index:02d}" for index in range(1, 11))


@dataclass(frozen=True)
class OuterFold:
    task: str
    repeat_id: str
    repeat_seed: str
    fold_id: str
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    test_strata: Mapping[str, str]


def rank(key: object, seed: str) -> str:
    return hashlib.sha256(f"{seed}::{key}".encode("utf-8")).hexdigest().upper()


def balanced_group_assignments(frame: pd.DataFrame, group_field: str, folds: int, seed: str) -> pd.Series:
    sizes = frame.groupby(group_field, sort=False).size()
    ordered = sorted(sizes.items(), key=lambda item: (-item[1], rank(item[0], seed)))
    loads = [{"fold": index, "rows": 0, "groups": 0} for index in range(folds)]
    group_to_fold = {}
    for group, size in ordered:
        target = sorted(loads, key=lambda item: (item["rows"], item["groups"], item["fold"]))[0]
        group_to_fold[group] = target["fold"]
        target["rows"] += int(size)
        target["groups"] += 1
    return frame[group_field].map(group_to_fold).astype(int)


def e1_inner_assignments(frame: pd.DataFrame, folds: int, seed: str) -> pd.Series:
    """Assign eligible base mixes; single-mix campaigns remain training-only."""
    output = pd.Series(-1, index=frame.index, dtype=int)
    loads = [{"fold": index, "rows": 0, "groups": 0} for index in range(folds)]
    campaign_entries = []
    for campaign, campaign_rows in frame.groupby("experimental_campaign_id", sort=False):
        sizes = campaign_rows.groupby("base_mix_id").size()
        if len(sizes) >= 2:
            campaign_entries.append((campaign, sizes, int(sizes.sum())))
    campaign_entries.sort(key=lambda item: (-item[2], rank(item[0], seed)))
    mix_to_fold: dict[str, int] = {}
    for campaign, sizes, _ in campaign_entries:
        mixes = sorted(sizes.items(), key=lambda item: (-item[1], rank(f"{campaign}::{item[0]}", seed)))
        first_fold = None
        for position, (mix, size) in enumerate(mixes):
            candidates = [load for load in loads if position != 1 or load["fold"] != first_fold]
            target = sorted(candidates, key=lambda item: (item["rows"], item["groups"], item["fold"]))[0]
            if position == 0:
                first_fold = target["fold"]
            mix_to_fold[mix] = target["fold"]
            target["rows"] += int(size)
            target["groups"] += 1
    mapped = frame["base_mix_id"].map(mix_to_fold)
    output.loc[mapped.notna()] = mapped.loc[mapped.notna()].astype(int)
    for campaign, subset in frame.loc[output >= 0].groupby("experimental_campaign_id"):
        if subset["base_mix_id"].map(mix_to_fold).nunique() < 2:
            raise AssertionError(f"E1 inner campaign is not represented in training: {campaign}")
    return output


def make_inner_assignments(frame: pd.DataFrame, task: str, seed: str) -> pd.Series:
    identifiers = frame[[
        "record_id", "base_mix_id", "experimental_campaign_id", "research_programme_id"
    ]].copy()
    if task == "E0":
        return balanced_group_assignments(identifiers, "record_id", 5, f"{seed}::INNER_E0")
    if task == "E1":
        return e1_inner_assignments(identifiers, 5, f"{seed}::INNER_E1")
    if task == "E3a":
        return balanced_group_assignments(identifiers, "experimental_campaign_id", 4, f"{seed}::INNER_E3A")
    if task == "E3b":
        return balanced_group_assignments(identifiers, "research_programme_id", 4, f"{seed}::INNER_E3B")
    raise ValueError(f"Unknown task: {task}")


def inner_score(frame: pd.DataFrame, predictions: np.ndarray, task: str, eligible: np.ndarray) -> float:
    group_field = "research_programme_id" if task == "E3b" else "experimental_campaign_id"
    scored = frame.loc[eligible, [group_field]].copy()
    scored["absolute_error"] = np.abs(frame.loc[eligible, core.TARGET].to_numpy(dtype=float) - predictions[eligible])
    return float(scored.groupby(group_field, sort=True)["absolute_error"].mean().mean())


def fit_with_convergence_guard(estimator, features, target):
    """Fit once and convert any optimization non-convergence into a hard failure."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted = estimator.fit(features, target)
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError(f"MODEL_DID_NOT_CONVERGE: {convergence[0]}")
    return fitted


def load_candidate_grid(path: Path) -> dict[str, list[dict[str, object]]]:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("status") != "PASS_EXACT_CANDIDATE_LISTS_FROZEN_NO_FITTING":
        raise core.RunBlocked("MODEL_CANDIDATE_GRID_NOT_FROZEN")
    return {
        model: [candidate for candidate in entry["candidates"]]
        for model, entry in document["grids"].items()
    }


def tune_model(
    outer_train: pd.DataFrame,
    task: str,
    panel: str,
    model: str,
    candidates: list[dict[str, object]],
    tuning_seed_label: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    assignments = make_inner_assignments(outer_train, task, tuning_seed_label)
    validation_mask = assignments.to_numpy() >= 0
    if not validation_mask.any() or assignments[assignments >= 0].nunique() < 2:
        raise RuntimeError(f"Infeasible inner split for {task}")
    numeric, categorical = core.panel_fields(panel)
    features = list(numeric + categorical)
    audit_rows = []
    for candidate in candidates:
        predictions = np.full(len(outer_train), np.nan, dtype=float)
        failure = ""
        try:
            for fold in sorted(assignments[assignments >= 0].unique()):
                inner_validation = assignments.to_numpy() == fold
                inner_training = ~inner_validation
                seed = core.seed_from_labels(tuning_seed_label, model, f"INNER_{fold}")
                pipeline = core.make_model_pipeline(model, candidate["parameters"], seed, panel)
                fit_with_convergence_guard(
                    pipeline,
                    outer_train.loc[inner_training, features],
                    outer_train.loc[inner_training, core.TARGET],
                )
                predictions[inner_validation] = pipeline.predict(outer_train.loc[inner_validation, features])
            score = inner_score(outer_train, predictions, task, validation_mask)
            if not np.isfinite(score):
                raise RuntimeError("Non-finite inner score")
        except Exception as exc:
            score = float("inf")
            failure = f"{type(exc).__name__}: {exc}"
        audit_rows.append({
            "candidate_id": candidate["candidate_id"],
            "candidate_hash": candidate["ordering_sha256"],
            "inner_macro_mae": score,
            "failure": failure,
        })
    successful = [row for row in audit_rows if np.isfinite(row["inner_macro_mae"])]
    if not successful:
        raise RuntimeError(f"Every {model} candidate failed under {task}/{panel}; definitive run aborted")
    selected_audit = sorted(successful, key=lambda row: (row["inner_macro_mae"], row["candidate_hash"]))[0]
    selected = next(candidate for candidate in candidates if candidate["candidate_id"] == selected_audit["candidate_id"])
    return selected, audit_rows


def validate_input_schema(data: pd.DataFrame) -> None:
    required = {
        "record_id", "base_mix_id", "publication_family_id", "experimental_campaign_id",
        "research_programme_id", core.TARGET,
        *core.P2_NUMERIC, *core.P2_CATEGORICAL,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Model-ready input lacks fields: {missing}")
    if data.empty:
        raise ValueError("Model-ready input is empty")
    if data["record_id"].isna().any() or data["record_id"].duplicated().any():
        raise ValueError("record_id must be complete and unique")
    for field in ("base_mix_id", "publication_family_id", "experimental_campaign_id", "research_programme_id"):
        if data[field].isna().any():
            raise ValueError(f"{field} contains missing values")
    if data[core.TARGET].isna().any():
        raise ValueError("Target contains missing values")


def outer_folds(data: pd.DataFrame, manifest: pd.DataFrame, repeated: pd.DataFrame) -> Iterator[OuterFold]:
    all_ids = set(data["record_id"])
    if set(manifest["record_id"]) != all_ids or set(repeated["record_id"]) != all_ids:
        raise ValueError("Model input and split manifests have different record IDs")
    for repeat_id, repeat_rows in repeated.groupby("repeat_id", sort=True):
        repeat_seed = repeat_rows["repeat_seed"].iloc[0]
        for task, fold_field in (("E0", "e0_fold"), ("E1", "e1_fold_or_exclusion")):
            task_rows = repeat_rows
            if task == "E1":
                task_rows = task_rows[task_rows["e1_inclusion_status"] == "E1_EVALUABLE_CAMPAIGN_REPRESENTED_IN_TRAINING"]
            task_ids = set(task_rows["record_id"])
            for fold_id in sorted(value for value in task_rows[fold_field].unique() if value.startswith(task)):
                test_rows = task_rows[task_rows[fold_field] == fold_id]
                test_ids = tuple(sorted(test_rows["record_id"]))
                train_ids = tuple(sorted(task_ids - set(test_ids)))
                strata = dict(zip(test_rows["record_id"], test_rows["e0_mechanism_stratum"], strict=True)) if task == "E0" else {}
                yield OuterFold(task, repeat_id, repeat_seed, fold_id, train_ids, test_ids, strata)
    for task, fold_field in (("E3a", "e3a_leave_one_campaign_fold"), ("E3b", "e3b_leave_one_programme_fold")):
        for fold_id in sorted(manifest[fold_field].unique()):
            test_ids = tuple(sorted(manifest.loc[manifest[fold_field] == fold_id, "record_id"]))
            train_ids = tuple(sorted(all_ids - set(test_ids)))
            yield OuterFold(task, "FIXED", "FIXED_TARGET_INDEPENDENT_PARTITION", fold_id, train_ids, test_ids, {})


def model_configurations() -> list[tuple[str, str]]:
    result = [("mean", "BASELINE"), ("median", "BASELINE"), ("age_log_linear", "BASELINE"), ("ridge", "P0")]
    for model in ("elastic_net", "extra_trees", "hist_gradient_boosting", "catboost"):
        result.extend((model, panel) for panel in ("P0", "P1", "P2"))
    return result


def refit_seed_labels(task: str, model: str, repeat_seed: str) -> tuple[str, ...]:
    if task in {"E3a", "E3b"} and model in STOCHASTIC_MODELS:
        return MODEL_SEED_LABELS
    return (repeat_seed,)


def baseline_predict(model: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    target = train[core.TARGET].to_numpy(dtype=float)
    if model == "mean":
        return np.full(len(test), float(target.mean()))
    if model == "median":
        return np.full(len(test), float(np.median(target)))
    if model == "age_log_linear":
        learner = LinearRegression()
        learner.fit(np.log1p(train[["curing_age_days"]].to_numpy(dtype=float)), target)
        return learner.predict(np.log1p(test[["curing_age_days"]].to_numpy(dtype=float)))
    raise ValueError(model)


def execute_benchmark(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    repeated: pd.DataFrame,
    grids: dict[str, list[dict[str, object]]],
    output_dir: Path,
) -> None:
    data = data.set_index("record_id", drop=False)
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    logs: list[str] = []
    for outer in outer_folds(data.reset_index(drop=True), manifest, repeated):
        train = data.loc[list(outer.train_ids)].reset_index(drop=True)
        test = data.loc[list(outer.test_ids)].reset_index(drop=True)
        distances = core.gower_min_distance_p0(train, test)
        for record_id, value in zip(test["record_id"], distances, strict=True):
            distance_rows.append({
                "task": outer.task, "repeat_id": outer.repeat_id, "outer_fold_id": outer.fold_id,
                "record_id": record_id, "gower_p0_min_train_distance": float(value),
            })
        for model, panel in model_configurations():
            selected = None
            tuning_seed = f"{outer.repeat_seed}::{outer.task}::{outer.fold_id}::{model}::{panel}::TUNING"
            if model not in {"mean", "median", "age_log_linear"}:
                selected, audit = tune_model(train, outer.task, panel, model, grids[model], tuning_seed)
                selection_rows.append({
                    "task": outer.task, "repeat_id": outer.repeat_id, "outer_fold_id": outer.fold_id,
                    "model": model, "panel": panel, "selected_candidate_id": selected["candidate_id"],
                    "selected_candidate_hash": selected["ordering_sha256"],
                    "selected_parameters_json": json.dumps(selected["parameters"], sort_keys=True, separators=(",", ":")),
                    "candidate_audit_json": json.dumps(audit, sort_keys=True, separators=(",", ":")),
                })
            for seed_label in refit_seed_labels(outer.task, model, outer.repeat_seed):
                seed = core.seed_from_labels(seed_label, model, outer.fold_id)
                if model in {"mean", "median", "age_log_linear"}:
                    predictions = baseline_predict(model, train, test)
                else:
                    numeric, categorical = core.panel_fields(panel)
                    features = list(numeric + categorical)
                    pipeline = core.make_model_pipeline(model, selected["parameters"], seed, panel)
                    fit_with_convergence_guard(pipeline, train[features], train[core.TARGET])
                    predictions = pipeline.predict(test[features])
                for row, prediction in zip(test.to_dict("records"), predictions, strict=True):
                    prediction_rows.append({
                        "task": outer.task, "repeat_id": outer.repeat_id, "repeat_seed": outer.repeat_seed,
                        "outer_fold_id": outer.fold_id, "model": model, "panel": panel,
                        "model_seed_label": seed_label, "model_seed": seed,
                        "record_id": row["record_id"], "base_mix_id": row["base_mix_id"],
                        "experimental_campaign_id": row["experimental_campaign_id"],
                        "research_programme_id": row["research_programme_id"],
                        "e0_mechanism_stratum": outer.test_strata.get(row["record_id"], "NOT_APPLICABLE"),
                        "y_true": float(row[core.TARGET]), "y_pred": float(prediction),
                    })
            logs.append(f"PASS {outer.task} {outer.repeat_id} {outer.fold_id} {model} {panel}")

    predictions = pd.DataFrame(prediction_rows)
    distances = pd.DataFrame(distance_rows).drop_duplicates()
    selections = pd.DataFrame(selection_rows)
    fold_keys = ["task", "repeat_id", "outer_fold_id", "model", "panel", "model_seed_label", "model_seed"]
    fold_metrics = []
    for keys, subset in predictions.groupby(fold_keys, sort=True):
        fold_metrics.append(dict(zip(fold_keys, keys, strict=True)) | {"n": len(subset)} | core.regression_metrics(subset["y_true"], subset["y_pred"]))
    fold_metrics = pd.DataFrame(fold_metrics)
    group_metrics = []
    run_keys = ["task", "repeat_id", "model", "panel", "model_seed_label", "model_seed"]
    for keys, subset in predictions.groupby(run_keys, sort=True):
        group_field = "research_programme_id" if keys[0] == "E3b" else "experimental_campaign_id"
        table = core.group_error_table(subset, group_field)
        for row in table.to_dict("records"):
            group_metrics.append(dict(zip(run_keys, keys, strict=True)) | {
                "group_field": group_field, "evaluation_group_id": row[group_field],
                "n": row["n"], "mae": row["mae"], "medae": row["medae"], "rmse": row["rmse"], "r2": row["r2"],
            })
    group_metrics = pd.DataFrame(group_metrics)
    uncertainty = build_uncertainty_document(predictions)

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output_dir / "outer_fold_predictions.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    fold_metrics.to_csv(output_dir / "outer_fold_metrics.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    group_metrics.to_csv(output_dir / "group_level_metrics.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    selections.to_csv(output_dir / "selected_hyperparameters.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    distances.to_csv(output_dir / "train_domain_distance.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    (output_dir / "uncertainty_and_influence.json").write_text(json.dumps(uncertainty, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "run_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")
    output_hashes = core.build_hash_bindings(sorted(output_dir.iterdir()))
    (output_dir / "run_manifest.json").write_text(json.dumps({
        "version": "LC3_BENCHMARK_OUTPUT_V1",
        "status": "COMPLETE_BENCHMARK_OUTPUTS",
        "python": sys.version,
        "output_sha256_before_run_manifest": output_hashes,
    }, indent=2) + "\n", encoding="utf-8")


def build_uncertainty_document(predictions: pd.DataFrame) -> dict[str, object]:
    """Aggregate each campaign from all out-of-fold predictions before pairing."""
    contrasts = []
    configurations = predictions[["model", "panel"]].drop_duplicates().itertuples(index=False, name=None)
    for model, panel in configurations:
        current = predictions[
            (predictions["model"] == model) & (predictions["panel"] == panel)
        ].assign(absolute_error=lambda frame: (frame["y_true"] - frame["y_pred"]).abs())
        e0 = (
            current[current["task"] == "E0"]
            .groupby("experimental_campaign_id", as_index=False)
            .agg(e0_mae=("absolute_error", "mean"), n=("record_id", "nunique"))
            .rename(columns={"experimental_campaign_id": "evaluation_group_id"})
        )
        e3 = (
            current[current["task"] == "E3a"]
            .groupby("experimental_campaign_id", as_index=False)
            .agg(e3a_mae=("absolute_error", "mean"))
            .rename(columns={"experimental_campaign_id": "evaluation_group_id"})
        )
        paired = e0.merge(e3, on="evaluation_group_id", how="inner", validate="one_to_one")
        if len(paired) < 2:
            continue
        bootstrap = core.campaign_cluster_bootstrap_contrast(paired, "e0_mae", "e3a_mae")
        influence = core.leave_one_campaign_influence(paired.set_index("evaluation_group_id"), "e0_mae", "e3a_mae")
        retained = paired[paired["n"] >= 3]
        contrasts.append({
            "model": model, "panel": panel,
            "campaign_count": len(paired),
            "mean_e0_campaign_macro_mae": float(paired["e0_mae"].mean()),
            "mean_e3a_campaign_macro_mae": float(paired["e3a_mae"].mean()),
            "e3a_to_e0_ratio": float(paired["e3a_mae"].mean() / paired["e0_mae"].mean()),
            "bootstrap": bootstrap,
            "small_campaign_exclusion_row_count_lt_3": {
                "campaign_count": int(len(retained)),
                "mean_difference": float((retained["e3a_mae"] - retained["e0_mae"]).mean())
                if not retained.empty
                else None,
            },
            "leave_one_campaign_influence": influence.to_dict("records"),
            "paired_campaign_errors": paired.to_dict("records"),
        })
    return {
        "version": "LC3_UNCERTAINTY_AND_INFLUENCE_V1",
        "campaigns_are_inferential_units": True,
        "seeds_or_repetitions_treated_as_independent": False,
        "contrasts": contrasts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--repeated-manifest", type=Path, required=True)
    parser.add_argument("--candidate-grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.execute == args.validate_only:
        raise SystemExit("Choose exactly one of --validate-only or --execute")
    data = pd.read_csv(args.input)
    validate_input_schema(data)
    manifest = pd.read_csv(args.split_manifest, dtype=str, keep_default_na=False)
    repeated = pd.read_csv(args.repeated_manifest, dtype=str, keep_default_na=False)
    list(outer_folds(data, manifest, repeated))
    grids = load_candidate_grid(args.candidate_grid)
    if args.validate_only:
        report = {
            "status": "PASS_RUNNER_VALIDATE_ONLY_NO_MODEL_FITTED",
            "rows": len(data),
            "outer_fold_instances": len(list(outer_folds(data, manifest, repeated))),
            "candidate_counts": {model: len(rows) for model, rows in grids.items()},
            "scientific_model_fitted": False,
            "scientific_results_generated": False,
        }
        if args.validation_report is not None:
            args.validation_report.parent.mkdir(parents=True, exist_ok=True)
            args.validation_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required with --execute")
    try:
        execute_benchmark(
            data,
            manifest,
            repeated,
            grids,
            args.output_dir,
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
