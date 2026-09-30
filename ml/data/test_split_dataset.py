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


def test_fixed_test_rows_are_preserved_and_train_validation_are_group_disjoint() -> None:
    frame = _manifest()
    fixed_test_ids = {"real-0", "real-1", "fake-0", "fake-1"}
    frame["split"] = frame["sample_id"].map(
        lambda sample_id: "test" if sample_id in fixed_test_ids else ""
    )

    result = create_splits(frame, seed=42)
    actual_test_ids = set(result.loc[result["split"].eq("test"), "sample_id"])
    assert actual_test_ids == fixed_test_ids

    train_groups = set(result.loc[result["split"].eq("train"), "source_group"])
    validation_groups = set(result.loc[result["split"].eq("validation"), "source_group"])
    assert train_groups.isdisjoint(validation_groups)
    assert len(result) == len(frame)
    assert all(
        set(group["label"]) == {"real", "fake"}
        for _, group in result.groupby("split")
    )


def test_split_rejects_too_few_groups_for_class_presence() -> None:
    tiny = _manifest().query("source_group in ['real-source-0', 'fake-source-0']")
    with pytest.raises(ValueError):
        create_splits(tiny, seed=42)
