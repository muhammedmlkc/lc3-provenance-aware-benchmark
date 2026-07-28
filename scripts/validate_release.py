"""Validate the code-only boundary of the public software release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.sha256"

FORBIDDEN_TOP_LEVEL = {"data", "metadata", "results", "outputs"}
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
}
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".tsv",
    ".xls",
    ".xlsx",
    ".doc",
    ".docx",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".pkl",
    ".pickle",
    ".joblib",
    ".parquet",
    ".feather",
}
TEXT_SUFFIXES = {
    "",
    ".py",
    ".mjs",
    ".md",
    ".txt",
    ".json",
    ".cff",
    ".sha256",
    ".gitignore",
    ".gitattributes",
}
SECRET_PATTERNS = (
    re.compile(
        r"(?i)(api[_-]?key|client[_-]?secret|password)"
        r"\s*[:=]\s*['\"][^'\"]+"
    ),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def included_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(ROOT).parts
        if any(part in IGNORED_DIRECTORY_NAMES for part in relative_parts):
            continue
        if path.suffix.lower() == ".pyc":
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def verify_manifest(files: list[Path]) -> int:
    require(MANIFEST.is_file(), "Release manifest is missing")
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        require("  " in line, "Malformed release-manifest line")
        expected, relative = line.split("  ", 1)
        require(
            re.fullmatch(r"[A-F0-9]{64}", expected) is not None,
            f"Invalid manifest digest: {relative}",
        )
        require(relative not in entries, f"Duplicate manifest path: {relative}")
        entries[relative] = expected

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in files
        if path != MANIFEST
    }
    require(
        set(entries) == actual,
        "Release manifest file list is incomplete or contains extra entries",
    )
    for relative, expected in entries.items():
        path = ROOT / Path(relative)
        require(path.is_file(), f"Manifest file missing: {relative}")
        require(sha256(path) == expected, f"Manifest hash mismatch: {relative}")
    return len(entries)


def validate_author_order() -> None:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    first = citation.find('family-names: "Demiral"')
    second = citation.find('family-names: "Malkoç"')
    require(first >= 0 and second >= 0 and first < second, "CITATION.cff author order is incorrect")
    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    require(
        "Nazım Çağatay Demiral and Muhammed Malkoç" in licence,
        "LICENSE author order is incorrect",
    )


def scan_tree(files: list[Path]) -> tuple[int, int]:
    text_count = 0
    for path in files:
        relative = path.relative_to(ROOT)
        relative_posix = relative.as_posix()
        require(
            not relative.parts or relative.parts[0].lower() not in FORBIDDEN_TOP_LEVEL,
            f"Forbidden public directory: {relative_posix}",
        )
        require(
            path.suffix.lower() not in FORBIDDEN_SUFFIXES,
            f"Data, result, document, or figure artifact included: {relative_posix}",
        )
        require(not path.is_symlink(), f"Symbolic link included: {relative_posix}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="strict")
        require(
            re.search(r"(?i)[a-z]:\\users\\", text) is None,
            f"Absolute Windows user path in: {relative_posix}",
        )
        require(
            re.search(r"(?i)/(home|users)/[^/\s]+/", text) is None,
            f"Absolute POSIX user path in: {relative_posix}",
        )
        require(
            not any(pattern.search(text) for pattern in SECRET_PATTERNS),
            f"Possible secret in: {relative_posix}",
        )
        text_count += 1
    return len(files), text_count


def main() -> dict[str, object]:
    files = included_files()
    public_files, text_files = scan_tree(files)
    validate_author_order()
    manifest_entries = verify_manifest(files)
    return {
        "status": "PASS_CODE_ONLY_RELEASE",
        "source_documents_distributed": 0,
        "source_derived_data_rows_distributed": 0,
        "study_result_files_distributed": 0,
        "publication_figures_distributed": 0,
        "manifest_hashes_verified": manifest_entries,
        "public_files_scanned": public_files,
        "text_files_scanned": text_files,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
