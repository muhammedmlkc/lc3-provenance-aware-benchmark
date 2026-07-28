"""Twelve pre-fit implementation tests using a fully synthetic fixture.

The suite creates all identifiers, hierarchy fields, and responses in a
temporary directory. It never reads a research dataset.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import core as benchmark


GENERATOR = REPO_ROOT / "scripts" / "generate_splits.mjs"
ASSIGNMENT_OUTPUTS = (
    "split_manifest.csv",
    "split_manifest_repeated.csv",
    "split_group_assignments.csv",
)

TEST_REQUIREMENTS = {
    "test_01_e1_base_mix_never_crosses_folds": "No E1 base mixture crosses train/test.",
    "test_02_e2_publication_never_crosses_folds": "No E2 publication family crosses train/test.",
    "test_03_e3a_campaign_never_crosses_folds": "No E3a campaign crosses train/test.",
    "test_04_e3b_programme_never_crosses_folds": "No E3b programme family crosses train/test.",
    "test_05_e1_campaign_is_represented_in_training": "Every evaluable E1 test campaign remains represented in training by another base mixture.",
    "test_06_preprocessor_state_is_training_only": "Preprocessors and encoders expose only training-fitted state.",
    "test_07_test_only_sentinel_never_enters_fitted_state": "A test-only sentinel category/value cannot enter fitted preprocessing state.",
    "test_08_split_generator_fails_on_target_access": "Split generation fails if code attempts to read the target field.",
    "test_09_target_permutation_does_not_change_assignments": "Target permutation does not change split assignments.",
    "test_10_synthetic_campaign_effect_exposes_validation_gap": "Synthetic campaign effects produce the expected row-wise versus campaign-held-out gap.",
    "test_11_hash_change_aborts_authorization": "A changed bound artifact is rejected by the hash-verification gate.",
    "test_12_repeated_execution_is_exact": "Repeated execution reproduces manifests and deterministic-model outputs exactly.",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run_generator(
    generator: Path,
    input_file: Path,
    programme_map: Path,
    output_dir: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        "node", str(generator),
        "--input", str(input_file),
        "--programme-map", str(programme_map),
        "--out", str(output_dir),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False, encoding="utf-8")


def write_synthetic_split_inputs(directory: Path) -> tuple[Path, Path]:
    """Create a small artificial hierarchy unrelated to the research corpus."""
    specifications = [
        ("SYN_PUBLICATION_01", 8, 4, "SYN_PROGRAM_SHARED"),
        ("SYN_PUBLICATION_02", 6, 3, "SYN_PROGRAM_SHARED"),
        ("SYN_PUBLICATION_03", 4, 2, "SYN_PROGRAM_03"),
        ("SYN_PUBLICATION_04", 5, 1, "SYN_PROGRAM_04"),
        ("SYN_PUBLICATION_05", 8, 4, "SYN_PROGRAM_05"),
        ("SYN_PUBLICATION_06", 6, 3, "SYN_PROGRAM_06"),
        ("SYN_PUBLICATION_07", 4, 2, "SYN_PROGRAM_07"),
        ("SYN_PUBLICATION_08", 3, 1, "SYN_PROGRAM_08"),
    ]
    source_rows: list[dict[str, str]] = []
    hierarchy_rows: list[dict[str, str]] = []
    record_index = 0
    for campaign_index, (publication, row_count, mix_count, programme) in enumerate(specifications, start=1):
        campaign = f"SYN_CAMPAIGN_{campaign_index:02d}"
        for local_index in range(row_count):
            record_index += 1
            record = f"SYN_RECORD_{record_index:03d}"
            mix = f"SYN_MIX_{campaign_index:02d}_{(local_index % mix_count) + 1:02d}"
            source_rows.append({
                "record_id": record,
                "base_mix_id": mix,
                "publication_family_id": publication,
                "compressive_strength_mpa": f"{1000.0 + record_index:.1f}",
            })
            hierarchy_rows.append({
                "record_id": record,
                "publication_family_id": publication,
                "experimental_campaign_id": campaign,
                "research_programme_id": programme,
            })

    source = directory / "synthetic_split_input.csv"
    hierarchy = directory / "synthetic_programme_map.csv"
    pd.DataFrame(source_rows).to_csv(source, index=False, encoding="utf-8-sig", lineterminator="\n")
    pd.DataFrame(hierarchy_rows).to_csv(hierarchy, index=False, encoding="utf-8-sig", lineterminator="\n")
    return source, hierarchy


def make_synthetic_model() -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), ["irrelevant_numeric"]),
        ("campaign", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["campaign"]),
    ])
    return Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=1e-9))])


class MandatoryImplementationTests(unittest.TestCase):
    synthetic_diagnostics: dict[str, float] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_context = tempfile.TemporaryDirectory(prefix="benchmark_implementation_tests_")
        cls.temp = Path(cls.temp_context.name)
        cls.source_input, cls.programme_map = write_synthetic_split_inputs(cls.temp)
        cls.original_a = cls.temp / "original_a"
        cls.original_b = cls.temp / "original_b"
        cls.permuted_dir = cls.temp / "target_permuted"
        for directory in (cls.original_a, cls.original_b, cls.permuted_dir):
            directory.mkdir()

        cls.run_original_a = run_generator(GENERATOR, cls.source_input, cls.programme_map, cls.original_a)
        cls.run_original_b = run_generator(GENERATOR, cls.source_input, cls.programme_map, cls.original_b)
        if cls.run_original_a.returncode != 0:
            raise RuntimeError(cls.run_original_a.stderr or cls.run_original_a.stdout)
        cls.manifest = pd.read_csv(cls.original_a / "split_manifest.csv", dtype=str, keep_default_na=False)
        cls.repeated = pd.read_csv(cls.original_a / "split_manifest_repeated.csv", dtype=str, keep_default_na=False)

        with cls.source_input.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            columns = list(rows[0].keys())
        strengths = [row[benchmark.TARGET] for row in rows][::-1]
        for row, permuted in zip(rows, strengths, strict=True):
            row[benchmark.TARGET] = permuted
        cls.permuted_input = cls.temp / "headline_target_permuted.csv"
        with cls.permuted_input.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        cls.run_permuted = run_generator(GENERATOR, cls.permuted_input, cls.programme_map, cls.permuted_dir)

        mutated_text = GENERATOR.read_text(encoding="utf-8").replace(
            "// TARGET_GUARD_MUTATION_TEST_MARKER",
            "// TARGET_GUARD_MUTATION_TEST_MARKER\n  void row.compressive_strength_mpa;",
            1,
        )
        cls.mutated_generator = cls.temp / "split_generator_forbidden_probe.mjs"
        cls.mutated_generator.write_text(mutated_text, encoding="utf-8")
        cls.mutated_dir = cls.temp / "mutated"
        cls.mutated_dir.mkdir()
        cls.run_mutated = run_generator(cls.mutated_generator, cls.source_input, cls.programme_map, cls.mutated_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_context.cleanup()

    def test_01_e1_base_mix_never_crosses_folds(self) -> None:
        evaluable = self.repeated[self.repeated["e1_inclusion_status"] == "E1_EVALUABLE_CAMPAIGN_REPRESENTED_IN_TRAINING"]
        crossing = evaluable.groupby(["repeat_id", "base_mix_id"])["e1_fold_or_exclusion"].nunique()
        self.assertTrue((crossing == 1).all())

    def test_02_e2_publication_never_crosses_folds(self) -> None:
        benchmark.assert_group_integrity(self.manifest, "publication_family_id", "e2_leave_one_publication_fold")

    def test_03_e3a_campaign_never_crosses_folds(self) -> None:
        benchmark.assert_group_integrity(self.manifest, "experimental_campaign_id", "e3a_leave_one_campaign_fold")

    def test_04_e3b_programme_never_crosses_folds(self) -> None:
        benchmark.assert_group_integrity(self.manifest, "research_programme_id", "e3b_leave_one_programme_fold")

    def test_05_e1_campaign_is_represented_in_training(self) -> None:
        evaluable = self.repeated[self.repeated["e1_inclusion_status"] == "E1_EVALUABLE_CAMPAIGN_REPRESENTED_IN_TRAINING"]
        folds = evaluable.groupby(["repeat_id", "experimental_campaign_id"])["e1_fold_or_exclusion"].nunique()
        self.assertTrue((folds >= 2).all(), "At least one evaluable campaign occupies only one E1 fold")

    def test_06_preprocessor_state_is_training_only(self) -> None:
        train = pd.DataFrame({"numeric": [1.0, 2.0, np.nan, 4.0], "category": ["A", "A", "B", "B"]})
        test = pd.DataFrame({"numeric": [999999.0], "category": ["TEST_ONLY"]})
        preprocessor = benchmark.make_fold_local_preprocessor(["numeric"], ["category"])
        preprocessor.fit(train)
        before = benchmark.fitted_preprocessor_state(preprocessor)
        preprocessor.transform(test)
        after = benchmark.fitted_preprocessor_state(preprocessor)
        self.assertEqual(before, after)
        self.assertAlmostEqual(before["numeric_imputer_statistics"][0], 2.0)
        self.assertEqual(before["categorical_levels"][0], ["A", "B"])

    def test_07_test_only_sentinel_never_enters_fitted_state(self) -> None:
        train = pd.DataFrame({"numeric": [1.0, 2.0, 3.0], "category": ["A", "A", "B"]})
        test = pd.DataFrame({"numeric": [1.0e12], "category": ["SENTINEL_TEST_ONLY"]})
        preprocessor = benchmark.make_fold_local_preprocessor(["numeric"], ["category"])
        preprocessor.fit(train)
        preprocessor.transform(test)
        state = benchmark.fitted_preprocessor_state(preprocessor)
        self.assertNotIn("SENTINEL_TEST_ONLY", state["categorical_levels"][0])
        self.assertEqual(state["numeric_imputer_statistics"][0], 2.0)
        self.assertLess(max(abs(value) for value in state["numeric_scaler_mean"]), 10.0)

    def test_08_split_generator_fails_on_target_access(self) -> None:
        combined = self.run_mutated.stdout + self.run_mutated.stderr
        self.assertNotEqual(self.run_mutated.returncode, 0)
        self.assertIn("TARGET_ACCESS_PROHIBITED", combined)

    def test_09_target_permutation_does_not_change_assignments(self) -> None:
        self.assertEqual(self.run_original_a.returncode, 0, self.run_original_a.stderr)
        self.assertEqual(self.run_permuted.returncode, 0, self.run_permuted.stderr)
        for filename in ASSIGNMENT_OUTPUTS:
            self.assertEqual(sha256(self.original_a / filename), sha256(self.permuted_dir / filename), filename)

    def test_10_synthetic_campaign_effect_exposes_validation_gap(self) -> None:
        campaigns = [f"C{i:02d}" for i in range(8)]
        effects = np.asarray([-40.0, -30.0, -20.0, -10.0, 10.0, 20.0, 30.0, 40.0])
        records = []
        for campaign_index, campaign in enumerate(campaigns):
            for row_index in range(20):
                records.append({
                    "campaign": campaign,
                    "irrelevant_numeric": float((row_index * 7 + campaign_index) % 11),
                    "within_campaign_index": row_index,
                    "target": effects[campaign_index] + ((row_index % 3) - 1) * 0.01,
                })
        data = pd.DataFrame(records)
        row_predictions = np.empty(len(data), dtype=float)
        for fold in range(5):
            test_mask = data["within_campaign_index"] % 5 == fold
            model = make_synthetic_model()
            model.fit(data.loc[~test_mask, ["irrelevant_numeric", "campaign"]], data.loc[~test_mask, "target"])
            row_predictions[test_mask] = model.predict(data.loc[test_mask, ["irrelevant_numeric", "campaign"]])
        campaign_predictions = np.empty(len(data), dtype=float)
        for campaign in campaigns:
            test_mask = data["campaign"] == campaign
            model = make_synthetic_model()
            model.fit(data.loc[~test_mask, ["irrelevant_numeric", "campaign"]], data.loc[~test_mask, "target"])
            campaign_predictions[test_mask] = model.predict(data.loc[test_mask, ["irrelevant_numeric", "campaign"]])
        row_mae = float(np.mean(np.abs(data["target"].to_numpy() - row_predictions)))
        campaign_mae = float(np.mean(np.abs(data["target"].to_numpy() - campaign_predictions)))
        type(self).synthetic_diagnostics = {
            "rowwise_mae": row_mae,
            "campaign_held_out_mae": campaign_mae,
            "gap_ratio": campaign_mae / row_mae,
        }
        self.assertGreater(campaign_mae, row_mae * 100.0)

    def test_11_hash_change_aborts_authorization(self) -> None:
        original = self.temp / "hash_original.txt"
        changed = self.temp / "hash_changed.txt"
        original.write_text("frozen\n", encoding="utf-8")
        changed.write_text("frozen\n", encoding="utf-8")
        expected = benchmark.build_hash_bindings([original])
        changed.write_text("mutated\n", encoding="utf-8")
        renamed_expected = {changed.name: next(iter(expected.values()))}
        with self.assertRaises(benchmark.RunBlocked):
            benchmark.verify_hash_bindings(renamed_expected, [changed])

    def test_12_repeated_execution_is_exact(self) -> None:
        self.assertEqual(self.run_original_a.returncode, 0, self.run_original_a.stderr)
        self.assertEqual(self.run_original_b.returncode, 0, self.run_original_b.stderr)
        for filename in ASSIGNMENT_OUTPUTS:
            self.assertEqual(sha256(self.original_a / filename), sha256(self.original_b / filename), filename)
        x = pd.DataFrame({"irrelevant_numeric": np.arange(12, dtype=float), "campaign": ["A", "B", "C"] * 4})
        y = np.asarray([1.0, 5.0, 9.0] * 4) + np.arange(12) * 0.01
        first = make_synthetic_model().fit(x, y).predict(x)
        second = make_synthetic_model().fit(x, y).predict(x)
        self.assertTrue(np.array_equal(first, second))


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[dict[str, str]] = []

    @staticmethod
    def method_name(test: unittest.TestCase) -> str:
        return test.id().rsplit(".", 1)[-1]

    def addSuccess(self, test):
        super().addSuccess(test)
        method = self.method_name(test)
        self.records.append({"test": method, "requirement": TEST_REQUIREMENTS[method], "status": "PASS"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        method = self.method_name(test)
        self.records.append({"test": method, "requirement": TEST_REQUIREMENTS[method], "status": "FAIL", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        method = self.method_name(test)
        self.records.append({"test": method, "requirement": TEST_REQUIREMENTS[method], "status": "ERROR", "detail": self._exc_info_to_string(err, test)})


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MandatoryImplementationTests)
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordingResult)
    result: RecordingResult = runner.run(suite)
    log_text = stream.getvalue()
    records = sorted(result.records, key=lambda item: item["test"])
    report = {
        "version": "LC3_IMPLEMENTATION_TESTS_V1",
        "status": "PASS_12_OF_12_ARTIFICIAL_INPUT_ONLY" if result.wasSuccessful() and len(records) == 12 else "FAIL_IMPLEMENTATION_TESTS",
        "test_count": len(records),
        "passed": sum(record["status"] == "PASS" for record in records),
        "failed_or_errored": sum(record["status"] != "PASS" for record in records),
        "research_data_read": False,
        "benchmark_model_fitted_to_research_data": False,
        "synthetic_diagnostics": MandatoryImplementationTests.synthetic_diagnostics,
        "tests": records,
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": Path(sys.executable).name,
            "node": subprocess.run(["node", "--version"], text=True, capture_output=True, check=True).stdout.strip(),
            "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
    }
    print(log_text)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
