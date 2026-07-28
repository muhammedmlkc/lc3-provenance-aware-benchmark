"""Build a SHA-256 manifest for every public-release file."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "RELEASE_MANIFEST.sha256"
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def included_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path == OUTPUT
            or any(
                part in IGNORED_DIRECTORY_NAMES
                for part in path.relative_to(ROOT).parts
            )
            or path.suffix.lower() == ".pyc"
        ):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> None:
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in included_files()]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} hashes to {OUTPUT.name}")


if __name__ == "__main__":
    main()
