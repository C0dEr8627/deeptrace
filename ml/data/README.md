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
| `source_group` | Grouping key used for split assignment |
| `split` | Optional preassigned split; currently used to reserve supplied test-list entries |

For extracted frames, all frames from one source video must share the same `source_group`. For paired real/fake images derived from the same source/identity, use the same grouping key when the dataset supports it. If source grouping metadata is unavailable, explicitly document that limitation; do not claim leakage-safe evaluation.

## Celeb-DF v2 local manifest

The builder scans the three expected top-level video categories and uses the category to assign labels: `Celeb-real` and `YouTube-real` are `real`; `Celeb-synthesis` is `fake`. It reads `List_of_testing_videos.txt` and marks those exact paths as `test`. The test list's leading 0/1 value is checked against the category mapping.

Example for Windows PowerShell (run from the repository root; adjust the dataset path as needed):

```powershell
$dataset = "C:\Users\UMAR\OneDrive\Desktop\celeb\Celeb-DF-v2"
$output = "C:\Users\UMAR\OneDrive\Desktop\deeptrace-local\celebdf_v2_manifest.csv"

python ml/data/build_manifest.py --dataset-root $dataset --output $output
```

The builder currently assigns filename-derived groups:
- `Celeb-real`: clips sharing a filename identity ID are grouped together.
- `Celeb-synthesis`: clips sharing the same unordered pair of filename IDs are grouped together.
- `YouTube-real`: each filename is its own group because no group identifier is apparent in the filename.

These are pragmatic filename-derived heuristics, not verified source-video provenance. The connected identity graph in Celeb-synthesis is highly interconnected, so these groups do not ensure identity-disjoint splits. Document this limitation in experiments and reports. The supplied test list is preserved exactly; consequently, filename-derived groups may cross the fixed test boundary. Train and validation groups are kept disjoint.

## Validation

From the repository root, with the backend environment active:

```powershell
$env:PYTHONPATH = "."
python ml/data/validate_manifest.py --manifest $output
```

Validation checks required columns, labels, duplicate sample IDs, empty grouping keys, and file existence. The script does not infer or invent source identities.

## Split the manifest

```powershell
$env:PYTHONPATH = "."
python ml/data/split_dataset.py --manifest $output --output "C:\Users\UMAR\OneDrive\Desktop\deeptrace-local\celebdf_v2_split.csv" --seed 42
```

When the manifest contains preassigned `test` rows, the splitter preserves them exactly and group-splits the remaining pool into train and validation. Validation targets approximately 10% of the full dataset; because the official test list is 518 videos (about 7.9% of the 6,529-video collection), training will be approximately 82%, validation 10%, and test 7.9%, with variation from indivisible groups. Both classes are required in every split. Train and validation groups must be disjoint; the fixed test group boundary is not guaranteed to be disjoint and must be reported as a limitation.

For manifests without a preassigned test column, the existing group-aware 80/10/10 splitter remains available. Inspect and record per-split class counts before training. Never split extracted frames independently at frame level.

Dataset files and generated manifests containing local paths should remain local and be excluded from Git.

## Extract frames from the split videos

Extract frames only after the video-level split is finalized. The extractor selects up to five evenly spaced frames per video and writes them under dataset/split/label folders. It creates a frame-level manifest that retains the source video ID, label, dataset, source group, and split. Extraction failures are written to a separate CSV; review them before training.

The extractor uses OpenCV, NumPy, and pandas. With the project environment active, install OpenCV if it is not already available:

```powershell
python -m pip install opencv-python
```

Run a small smoke test first (the first 10 rows in the split manifest):

```powershell
$local = "C:\Users\UMAR\OneDrive\Desktop\deeptrace-local"
python ml/data/extract_frames.py --manifest "$local\celebdf_v2_split.csv" --output-root "$local\frames" --output-manifest "$local\celebdf_v2_frames.csv" --frames-per-video 5 --limit-videos 10
```

Inspect several saved JPEGs and the generated frame manifest and failure report. If they look correct, run the same command without `--limit-videos 10` to process all videos. The extractor does not split frames, rebalance classes, or modify the source split. Re-running overwrites selected frame files and regenerates the output manifests.

Run the helper tests from the repository root:

```powershell
python -m unittest discover -s ml/data -p "test_extract_frames.py"
```

Expected maximum frame count for the full Celeb-DF v2 manifest is 32,645 (6,529 videos × 5 frames); the actual count can be lower when videos cannot be decoded or contain fewer frames. Keep the frames and CSVs containing local absolute paths outside Git.
