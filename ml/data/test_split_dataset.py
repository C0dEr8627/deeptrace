import pandas as pd
import pytest

from split_dataset import create_splits


def _manifest() -> pd.DataFrame:
    rows = []
    for label in ("real", "fake"):
        for group in range(30):
            rows.append(
                {
                    "sample_id": f"{label}-{group}",
                    "path": f"{label}/{group}.jpg",
                    "label": label,
                    "dataset": "fixture",
                    "source_group": f"{label}-source-{group}",
                }
            )
    return pd.DataFrame(rows)


def test_split_is_reproducible_and_group_disjoint() -> None:
    first = create_splits(_manifest(), seed=42)
    second = create_splits(_manifest(), seed=42)
    assert first[["sample_id", "split"]].equals(second[["sample_id", "split"]])
    assert first.groupby("source_group")["split"].nunique().max() == 1
    assert set(first["split"]) == {"train", "validation", "test"}
    assert all(set(group["label"]) == {"real", "fake"} for _, group in first.groupby("split"))


def test_split_rejects_too_few_groups_for_class_presence() -> None:
    tiny = _manifest().query("source_group in ['real-source-0', 'fake-source-0']")
    with pytest.raises(ValueError, match="both classes"):
        create_splits(tiny, seed=42)
