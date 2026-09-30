"""Create reproducible train/validation/test splits without source-group leakage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from validate_manifest import validate_manifest


def _group_split(
    frame: pd.DataFrame, *, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    left_idx, right_idx = next(
        splitter.split(frame, y=frame["label"], groups=frame["source_group"])
    )
    return frame.iloc[left_idx].copy(), frame.iloc[right_idx].copy()


def create_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign groups to 80/10/10 splits and reject unusable class distributions."""
    train_val, test = _group_split(frame, test_size=0.10, seed=seed)
    train, validation = _group_split(train_val, test_size=(1 / 9), seed=seed + 1)

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"
    result = pd.concat([train, validation, test]).sort_values("sample_id").reset_index(drop=True)

    required = {"real", "fake"}
    for split_name, subset in result.groupby("split"):
        labels = set(subset["label"])
        if labels != required:
            raise ValueError(
                f"Split '{split_name}' must contain both classes; found {sorted(labels)}. "
                "More source groups or a different dataset protocol may be required."
            )

    group_splits = result.groupby("source_group")["split"].nunique()
    if group_splits.gt(1).any():
        raise RuntimeError("Source-group leakage detected across splits")
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


if __name__ == "__main__":
    main()
