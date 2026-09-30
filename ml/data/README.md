# DeepTrace Dataset Preparation

This directory contains dataset-independent preparation utilities. Do not commit raw media, extracted frames, restricted dataset files, or local absolute paths.

## Dataset acquisition status

No dataset is bundled with this repository. Before downloading any candidate, record its official source, version, access approval, terms/license, citation, label definition, and whether redistribution is permitted. FaceForensics++ and Celeb-DF require following their respective access request and terms-of-use procedures; do not treat a public code repository or third-party mirror as permission to redistribute the media.

## Canonical manifest

Create a CSV manifest with these columns:

| Column | Meaning |
|---|---|
| `sample_id` | Stable unique identifier for the media sample |
| `path` | Path to the media file, relative to the project root or absolute local path |
| `label` | Exactly `real` or `fake` |
| `dataset` | Dataset/source identifier |
| `source_group` | Original video, identity, or other source grouping key used to prevent leakage |

For extracted frames, all frames from one source video must share the same `source_group`. For paired real/fake images derived from the same source/identity, use the same grouping key when the dataset supports it. If source grouping metadata is unavailable, explicitly document that limitation; do not claim leakage-safe evaluation.

## Validation

From the repository root, with the backend environment active:

```powershell
$env:PYTHONPATH = "."
python ml/data/validate_manifest.py --manifest data/manifest.csv
```

Validation checks required columns, labels, duplicate sample IDs, empty grouping keys, and file existence. The script does not infer or invent source identities.

## Group-aware split

After validating the manifest:

```powershell
$env:PYTHONPATH = "."
python ml/data/split_dataset.py --manifest data/manifest.csv --output data/manifest_split.csv --seed 42
```

The splitter targets 80/10/10 train/validation/test proportions using group-aware splitting. Exact proportions can vary because groups are indivisible. It fails if a split is missing a class or if the resulting groups overlap. Inspect and record per-split class counts before training. Never split extracted frames independently at frame level.

Dataset files and generated manifests containing local paths should remain local and be excluded from Git.
