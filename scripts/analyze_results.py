"""Aggregate private benchmark outputs without embedding study results.

All paths are supplied at runtime. The script writes derived tables only to the
requested local output directory; no data or result is bundled with the public
software release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


BOOTSTRAP_SEED_LABEL = "CAMPAIGN_BOOTSTRAP_V1"
BOOTSTRAP_DRAWS = 10_000


def seed_from_label(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def regression_metrics(frame: pd.DataFrame) -> dict[str, float]:
    truth = frame["y_true"].to_numpy(dtype=float)
    pred = frame["y_pred"].to_numpy(dtype=float)
    error = truth - pred
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "pooled_mae": float(np.mean(np.abs(error))),
        "pooled_medae": float(np.median(np.abs(error))),
        "pooled_rmse": float(np.sqrt(np.mean(error**2))),
        "pooled_r2": float(1 - np.sum(error**2) / denominator)
        if denominator > 0
        else math.nan,
    }


def group_errors(frame: pd.DataFrame, group_field: str) -> pd.DataFrame:
    return (
        frame.assign(abs_error=(frame["y_true"] - frame["y_pred"]).abs())
        .groupby(group_field, as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            prediction_count=("abs_error", "size"),
            record_count=("record_id", "nunique"),
        )
        .rename(columns={group_field: "evaluation_group_id"})
    )


def macro_summary(frame: pd.DataFrame, group_field: str) -> dict[str, float]:
    groups = group_errors(frame, group_field)
    return {
        "macro_mae": float(groups["mae"].mean()),
        "worst_group_mae": float(groups["mae"].max()),
        "upper_quartile_group_mae": float(groups["mae"].quantile(0.75)),
        "group_count": int(len(groups)),
    }


def paired_bootstrap(differences: np.ndarray) -> tuple[float, float]:
    if len(differences) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed_from_label(BOOTSTRAP_SEED_LABEL))
    sampled = rng.choice(
        differences,
        size=(BOOTSTRAP_DRAWS, len(differences)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def sign_test_pvalue(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return math.nan
    return float(
        binomtest(
            int(np.sum(nonzero > 0)),
            len(nonzero),
            0.5,
            alternative="two-sided",
        ).pvalue
    )


def write_csv(frame: pd.DataFrame, output_dir: Path, name: str) -> None:
    frame.to_csv(
        output_dir / name,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = args.results_dir / "outer_fold_predictions.csv"
    selections_path = args.results_dir / "selected_hyperparameters.csv"
    predictions = pd.read_csv(predictions_path)
    required = {
        "task",
        "repeat_id",
        "model",
        "panel",
        "model_seed_label",
        "record_id",
        "experimental_campaign_id",
        "research_programme_id",
        "y_true",
        "y_pred",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction file lacks fields: {missing}")
    if predictions.empty:
        raise ValueError("Prediction file is empty")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    configurations = list(
        predictions[["model", "panel"]]
        .drop_duplicates()
        .sort_values(["model", "panel"])
        .itertuples(index=False, name=None)
    )

    task_rows: list[dict[str, object]] = []
    grouped_rows: list[pd.DataFrame] = []
    for task in sorted(predictions["task"].unique()):
        group_field = (
            "research_programme_id"
            if task == "E3b"
            else "experimental_campaign_id"
        )
        for model, panel in configurations:
            subset = predictions[
                (predictions["task"] == task)
                & (predictions["model"] == model)
                & (predictions["panel"] == panel)
            ]
            if subset.empty:
                continue
            groups = group_errors(subset, group_field)
            groups.insert(0, "panel", panel)
            groups.insert(0, "model", model)
            groups.insert(0, "task", task)
            grouped_rows.append(groups)
            task_rows.append(
                {
                    "task": task,
                    "model": model,
                    "panel": panel,
                    "prediction_rows": int(len(subset)),
                    "record_count": int(subset["record_id"].nunique()),
                    "repeat_or_seed_count": int(
                        subset[["repeat_id", "model_seed_label"]]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    **regression_metrics(subset),
                    **macro_summary(subset, group_field),
                }
            )

    task_summary = pd.DataFrame(task_rows)
    all_group_errors = pd.concat(grouped_rows, ignore_index=True)

    paired_tables: list[pd.DataFrame] = []
    contrast_rows: list[dict[str, object]] = []
    for model, panel in configurations:
        e0 = all_group_errors[
            (all_group_errors["task"] == "E0")
            & (all_group_errors["model"] == model)
            & (all_group_errors["panel"] == panel)
        ][["evaluation_group_id", "mae", "record_count"]].rename(
            columns={"mae": "e0_mae"}
        )
        e3a = all_group_errors[
            (all_group_errors["task"] == "E3a")
            & (all_group_errors["model"] == model)
            & (all_group_errors["panel"] == panel)
        ][["evaluation_group_id", "mae"]].rename(columns={"mae": "e3a_mae"})
        paired = e0.merge(e3a, on="evaluation_group_id", validate="one_to_one")
        if paired.empty:
            continue
        paired["difference_e3a_minus_e0"] = paired["e3a_mae"] - paired["e0_mae"]
        paired.insert(0, "panel", panel)
        paired.insert(0, "model", model)
        paired_tables.append(paired)
        differences = paired["difference_e3a_minus_e0"].to_numpy(dtype=float)
        low, high = paired_bootstrap(differences)
        omit_one = [
            float(np.delete(differences, index).mean())
            for index in range(len(differences))
            if len(differences) > 1
        ]
        contrast_rows.append(
            {
                "model": model,
                "panel": panel,
                "campaign_count": int(len(paired)),
                "e0_campaign_macro_mae": float(paired["e0_mae"].mean()),
                "e3a_campaign_macro_mae": float(paired["e3a_mae"].mean()),
                "e3a_minus_e0": float(differences.mean()),
                "e3a_to_e0_ratio": float(
                    paired["e3a_mae"].mean() / paired["e0_mae"].mean()
                ),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "campaigns_e3a_worse_than_e0": int(np.sum(differences > 0)),
                "campaigns_e3a_better_than_e0": int(np.sum(differences < 0)),
                "sign_test_two_sided_p": sign_test_pvalue(differences),
                "omit_one_min": min(omit_one) if omit_one else math.nan,
                "omit_one_max": max(omit_one) if omit_one else math.nan,
            }
        )

    contrasts = pd.DataFrame(contrast_rows)
    paired_errors = (
        pd.concat(paired_tables, ignore_index=True)
        if paired_tables
        else pd.DataFrame()
    )

    e1_rows: list[dict[str, object]] = []
    for model, panel in configurations:
        e0 = all_group_errors[
            (all_group_errors["task"] == "E0")
            & (all_group_errors["model"] == model)
            & (all_group_errors["panel"] == panel)
        ][["evaluation_group_id", "mae"]].rename(columns={"mae": "e0_mae"})
        e1 = all_group_errors[
            (all_group_errors["task"] == "E1")
            & (all_group_errors["model"] == model)
            & (all_group_errors["panel"] == panel)
        ][["evaluation_group_id", "mae"]].rename(columns={"mae": "e1_mae"})
        paired = e0.merge(e1, on="evaluation_group_id", validate="one_to_one")
        if paired.empty:
            continue
        difference = paired["e1_mae"] - paired["e0_mae"]
        e1_rows.append(
            {
                "model": model,
                "panel": panel,
                "campaign_count": int(len(paired)),
                "e0_matched_campaign_macro_mae": float(paired["e0_mae"].mean()),
                "e1_campaign_macro_mae": float(paired["e1_mae"].mean()),
                "e1_minus_e0_matched": float(difference.mean()),
                "campaigns_e1_worse_than_e0": int((difference > 0).sum()),
            }
        )
    e1_matched = pd.DataFrame(e1_rows)

    age_comparison_rows: list[dict[str, object]] = []
    if not paired_errors.empty:
        age = paired_errors[
            (paired_errors["model"] == "age_log_linear")
            & (paired_errors["panel"] == "BASELINE")
        ][["evaluation_group_id", "e3a_mae"]].rename(
            columns={"e3a_mae": "age_e3a_mae"}
        )
        for model, panel in configurations:
            current = paired_errors[
                (paired_errors["model"] == model)
                & (paired_errors["panel"] == panel)
            ][["evaluation_group_id", "e3a_mae"]]
            paired = current.merge(age, on="evaluation_group_id", validate="one_to_one")
            if paired.empty:
                continue
            difference = (paired["e3a_mae"] - paired["age_e3a_mae"]).to_numpy(dtype=float)
            low, high = paired_bootstrap(difference)
            age_comparison_rows.append(
                {
                    "model": model,
                    "panel": panel,
                    "campaign_count": int(len(paired)),
                    "model_minus_age_mae": float(difference.mean()),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "campaigns_model_better_than_age": int((difference < 0).sum()),
                }
            )
    age_comparison = pd.DataFrame(age_comparison_rows)

    panel_rows: list[dict[str, object]] = []
    e3a_summary = task_summary[task_summary["task"] == "E3a"]
    for model in sorted(e3a_summary["model"].unique()):
        current = e3a_summary[e3a_summary["model"] == model].set_index("panel")
        if "P0" not in current.index:
            continue
        for panel in ("P1", "P2"):
            if panel not in current.index:
                continue
            panel_rows.append(
                {
                    "model": model,
                    "comparison": f"{panel}-P0",
                    "delta_macro_mae": float(
                        current.loc[panel, "macro_mae"]
                        - current.loc["P0", "macro_mae"]
                    ),
                }
            )
    panel_sensitivity = pd.DataFrame(panel_rows)

    stochastic = predictions[
        (predictions["task"] == "E3a")
        & predictions["model"].isin(["extra_trees", "catboost"])
    ]
    seed_rows: list[dict[str, object]] = []
    for (model, panel, seed_label), subset in stochastic.groupby(
        ["model", "panel", "model_seed_label"]
    ):
        seed_rows.append(
            {
                "model": model,
                "panel": panel,
                "model_seed_label": seed_label,
                "campaign_macro_mae": macro_summary(
                    subset, "experimental_campaign_id"
                )["macro_mae"],
            }
        )
    seed_level = pd.DataFrame(seed_rows)
    seed_summary = (
        seed_level.groupby(["model", "panel"], as_index=False)
        .agg(
            seed_count=("model_seed_label", "nunique"),
            mean_macro_mae=("campaign_macro_mae", "mean"),
            sd_macro_mae=("campaign_macro_mae", "std"),
            min_macro_mae=("campaign_macro_mae", "min"),
            max_macro_mae=("campaign_macro_mae", "max"),
        )
        if not seed_level.empty
        else pd.DataFrame()
    )

    selection_frequency = pd.DataFrame()
    if selections_path.is_file():
        selections = pd.read_csv(selections_path)
        needed = {"task", "model", "panel", "selected_candidate_id"}
        if needed.issubset(selections.columns) and not selections.empty:
            selection_frequency = (
                selections.groupby(
                    ["task", "model", "panel", "selected_candidate_id"],
                    as_index=False,
                )
                .size()
                .rename(columns={"size": "selection_count"})
            )
            totals = selection_frequency.groupby(
                ["task", "model", "panel"]
            )["selection_count"].transform("sum")
            selection_frequency["selection_fraction"] = (
                selection_frequency["selection_count"] / totals
            )

    outputs = {
        "task_configuration_summary.csv": task_summary,
        "paired_e3a_vs_e0_contrasts.csv": contrasts,
        "paired_campaign_errors.csv": paired_errors,
        "e1_vs_e0_matched_campaigns.csv": e1_matched,
        "models_vs_age_baseline.csv": age_comparison,
        "panel_sensitivity.csv": panel_sensitivity,
        "stochastic_seed_level.csv": seed_level,
        "stochastic_seed_summary.csv": seed_summary,
        "hyperparameter_selection_frequency.csv": selection_frequency,
    }
    for name, frame in outputs.items():
        write_csv(frame, args.output_dir, name)

    output_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest().upper()
        for path in sorted(args.output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "version": "LC3_GENERIC_DERIVED_ANALYSIS_V1",
        "status": "COMPLETE_DERIVED_ANALYSIS",
        "bootstrap": {
            "seed_label": BOOTSTRAP_SEED_LABEL,
            "draws": BOOTSTRAP_DRAWS,
            "inferential_unit": "experimental_campaign",
        },
        "output_sha256_before_manifest": output_hashes,
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
