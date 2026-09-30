"""Create reproducible train/validation/test splits with optional fixed test rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from validate_manifest import validate_manifest


def _group_split(
    frame: pd.DataFrame, *, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    left_idx, right_idx = next(
        splitter.split(frame, y=frame["label"], groups=frame["source_group"])
    )
    return frame.iloc[left_idx].copy(), frame.iloc[right_idx].copy()


def _validate_classes(result: pd.DataFrame) -> None:
    required = {"real", "fake"}
    for split_name, subset in result.groupby("split"):
        labels = set(subset["label"])
        if labels != required:
            raise ValueError(
                f"Split '{split_name}' must contain both classes; found {sorted(labels)}."
            )


def create_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Split by groups; honor preassigned test rows when a split column is supplied.

    For a fixed test set, train/validation groups are kept disjoint. Test rows are
    preserved exactly, so their filename-derived groups may also occur in the
    training pool; callers must document that limitation.
    """
    required_columns = {"sample_id", "label", "source_group"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot split an empty manifest")

    if "split" in frame.columns and frame["split"].fillna("").astype(str).str.strip().ne("").any():
        assignments = frame["split"].fillna("").astype(str).str.strip()
        invalid = set(assignments) - {"", "test"}
        if invalid:
            raise ValueError(
                "Input split column may contain only blank or 'test' values; "
                f"found {sorted(invalid)}"
            )
        test_mask = assignments.eq("test")
        pool = frame.loc[~test_mask].copy()
        test = frame.loc[test_mask].copy()
        if test.empty or pool.empty:
            raise ValueError("Fixed-test manifest must contain test and unassigned rows")
        # Target validation at ~10% of the full dataset while preserving the fixed test set.
        validation_fraction = 0.10 / (len(pool) / len(frame))
        train, validation = _group_split(
            pool, test_size=validation_fraction, seed=seed
        )
        train["split"] = "train"
        validation["split"] = "validation"
        test["split"] = "test"
        result = pd.concat([train, validation, test], ignore_index=True)
        # Fixed test list takes precedence. Only assert disjointness within train/val.
        train_groups = set(train["source_group"])
        validation_groups = set(validation["source_group"])
        if train_groups & validation_groups:
            raise RuntimeError("Source-group leakage detected between train and validation")
    else:
        train_val, test = _group_split(frame, test_size=0.10, seed=seed)
        train, validation = _group_split(train_val, test_size=(1 / 9), seed=seed + 1)
        train["split"] = "train"
        validation["split"] = "validation"
        test["split"] = "test"
        result = pd.concat([train, validation, test], ignore_index=True)
        group_splits = result.groupby("source_group")["split"].nunique()
        if group_splits.gt(1).any():
            raise RuntimeError("Source-group leakage detected across splits")

    result = result.sort_values("sample_id").reset_index(drop=True)
    _validate_classes(result)
    if result["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs detected after splitting")
    if len(result) != len(frame):
        raise RuntimeError("Split output row count differs from input manifest")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    frame = validate_manifest(args.manifest, args.project_root)
    result = create_splits(frame, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("Split counts:")
    print(pd.crosstab(result["split"], result["label"]).to_string())
    print(f"Wrote split manifest: {args.output}")
    if "split" in frame.columns and frame["split"].eq("test").any():
        print("Note: supplied test rows were preserved; source-group separation from test is not guaranteed.")


if __name__ == "__main__":
    main()
