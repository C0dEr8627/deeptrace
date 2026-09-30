"""Build a local Celeb-DF v2 video manifest with its supplied test list."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

CATEGORIES = {
    "Celeb-real": "real",
    "YouTube-real": "real",
    "Celeb-synthesis": "fake",
}
REAL_PATTERN = re.compile(r"^(id\d+)_")
FAKE_PATTERN = re.compile(r"^(id\d+)_(id\d+)_\d+$")


def source_group_for(category: str, stem: str) -> str:
    """Return a filename-derived grouping heuristic, not verified provenance."""
    if category == "Celeb-real":
        match = REAL_PATTERN.match(stem)
        if not match:
            raise ValueError(f"Unexpected Celeb-real filename: {stem}")
        return f"Celeb-real:{match.group(1)}"

    if category == "Celeb-synthesis":
        match = FAKE_PATTERN.match(stem)
        if not match:
            raise ValueError(f"Unexpected Celeb-synthesis filename: {stem}")
        # Pair is treated as unordered so reversed pair names share a group.
        pair = sorted(match.groups())
        return f"Celeb-synthesis:{pair[0]}_{pair[1]}"

    if category == "YouTube-real":
        return f"YouTube-real:{stem}"

    raise ValueError(f"Unsupported category: {category}")


def build_manifest(dataset_root: Path, testing_list: Path) -> pd.DataFrame:
    dataset_root = dataset_root.expanduser().resolve()
    testing_list = testing_list.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    if not testing_list.is_file():
        raise FileNotFoundError(f"Testing list not found: {testing_list}")

    reserved: set[str] = set()
    for line_number, raw in enumerate(testing_list.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or parts[0] not in {"0", "1"}:
            raise ValueError(f"Malformed testing-list line {line_number}: {raw}")
        relative = parts[1].replace("\\", "/").lstrip("./")
        category = relative.split("/", 1)[0]
        if category not in CATEGORIES or "/" not in relative:
            raise ValueError(f"Unexpected testing-list path on line {line_number}: {raw}")
        expected_label = "fake" if parts[0] == "0" else "real"
        if CATEGORIES[category] != expected_label:
            raise ValueError(f"Testing-list label/category mismatch on line {line_number}: {raw}")
        resolved = (dataset_root / Path(relative)).resolve()
        if not resolved.is_relative_to(dataset_root):
            raise ValueError(f"Testing-list path escapes dataset root: {relative}")
        if not resolved.is_file():
            raise FileNotFoundError(f"Testing-list media not found: {relative}")
        normalized = resolved.relative_to(dataset_root).as_posix()
        if normalized in reserved:
            raise ValueError(f"Duplicate testing-list path: {normalized}")
        reserved.add(normalized)

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for category, label in CATEGORIES.items():
        folder = dataset_root / category
        if not folder.is_dir():
            raise FileNotFoundError(f"Required category directory not found: {folder}")
        for media_path in sorted(folder.glob("*.mp4")):
            relative = media_path.relative_to(dataset_root).as_posix()
            sample_id = f"celebdf_v2/{category}/{media_path.stem}"
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            seen_ids.add(sample_id)
            rows.append({
                "sample_id": sample_id,
                "path": str(media_path.resolve()),
                "label": label,
                "dataset": "celebdf_v2",
                "source_group": source_group_for(category, media_path.stem),
                "split": "test" if relative in reserved else "",
            })

    frame = pd.DataFrame(rows, columns=[
        "sample_id", "path", "label", "dataset", "source_group", "split"
    ])
    found_reserved = {
        Path(value).resolve().relative_to(dataset_root).as_posix()
        for value in frame.loc[frame["split"].eq("test"), "path"]
    }
    if found_reserved != reserved:
        missing = sorted(reserved - found_reserved)
        raise ValueError(f"Testing-list entries not found in scanned categories: {missing[:10]}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--testing-list", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    testing_list = args.testing_list or args.dataset_root / "List_of_testing_videos.txt"
    frame = build_manifest(args.dataset_root, testing_list)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print("Manifest counts:")
    print(pd.crosstab(frame["label"], frame["split"].replace("", "unassigned")).to_string())
    print(f"Rows: {len(frame)}")
    print(f"Filename-derived groups: {frame['source_group'].nunique()}")
    print(f"Wrote manifest: {args.output}")
    print("Note: source_group is a filename-derived heuristic, not verified source provenance.")


if __name__ == "__main__":
    main()
