"""Validate a DeepTrace canonical media manifest without inferring metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"sample_id", "path", "label", "dataset", "source_group"}
VALID_LABELS = {"real", "fake"}


def validate_manifest(manifest_path: Path, project_root: Path) -> pd.DataFrame:
    """Load and validate the manifest; return it unchanged when valid."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Manifest contains no samples")

    for column in REQUIRED_COLUMNS:
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"Column '{column}' contains empty values")

    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample_id values: {duplicates[:5]}")

    labels = set(frame["label"].str.lower())
    if not labels.issubset(VALID_LABELS):
        raise ValueError(f"Labels must be real/fake; found: {sorted(labels)}")
    frame["label"] = frame["label"].str.lower()

    missing_files = []
    for value in frame["path"]:
        path = Path(value)
        resolved = path if path.is_absolute() else project_root / path
        if not resolved.is_file():
            missing_files.append(value)
    if missing_files:
        raise ValueError(f"Media files not found ({len(missing_files)}): {missing_files[:5]}")

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    frame = validate_manifest(args.manifest, args.project_root)
    print(f"Valid manifest: {len(frame)} samples")
    print(frame["label"].value_counts().to_string())
    print(f"Unique source groups: {frame['source_group'].nunique()}")


if __name__ == "__main__":
    main()
