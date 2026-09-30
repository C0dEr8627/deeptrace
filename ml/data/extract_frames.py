"""Extract a fixed number of evenly spaced frames from a split video manifest.

All output media and manifests are intended to remain local and outside Git.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"sample_id", "path", "label", "dataset", "source_group", "split"}
VALID_SPLITS = {"train", "validation", "test"}


def safe_id(value: str) -> str:
    """Convert a manifest sample ID into a filesystem-safe stable stem."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(value)).strip("._-")


def frame_indices(frame_count: int, count: int) -> list[int]:
    """Return unique, evenly spaced frame indices including video endpoints."""
    if frame_count <= 0:
        return []
    if count <= 0:
        raise ValueError("count must be positive")
    return sorted({int(round(x)) for x in np.linspace(0, frame_count - 1, min(count, frame_count))})


def extract_video(
    video_path: Path,
    output_dir: Path,
    sample_id: str,
    requested_frames: int,
    jpeg_quality: int = 95,
) -> tuple[list[dict[str, object]], str | None]:
    """Extract selected frames and return frame manifest rows or an error."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return [], "video_open_failed"

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    indices = frame_indices(frame_count, requested_frames)
    if not indices:
        capture.release()
        return [], "invalid_frame_count"

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_id(sample_id)
    rows: list[dict[str, object]] = []
    failed_indices: list[int] = []
    for frame_number, frame_index in enumerate(indices, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            failed_indices.append(frame_index)
            continue
        filename = f"{stem}__frame_{frame_number:02d}.jpg"
        destination = output_dir / filename
        saved = cv2.imwrite(
            str(destination), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not saved:
            failed_indices.append(frame_index)
            continue
        timestamp = frame_index / fps if fps > 0 else None
        rows.append({
            "frame_id": f"{sample_id}#frame_{frame_number:02d}",
            "frame_path": str(destination.resolve()),
            "video_sample_id": sample_id,
            "video_path": str(video_path.resolve()),
            "label": None,
            "dataset": None,
            "source_group": None,
            "split": None,
            "frame_index": frame_index,
            "timestamp_sec": timestamp,
        })

    capture.release()
    error = f"frame_read_or_write_failed:{','.join(map(str, failed_indices))}" if failed_indices else None
    return rows, error


def extract_manifest(
    manifest_path: Path,
    output_root: Path,
    output_manifest: Path,
    frames_per_video: int = 5,
    limit_videos: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract frames from a video split manifest without changing split membership."""
    videos = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(videos.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if videos.empty:
        raise ValueError("Video manifest is empty")
    if videos["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in video manifest")
    if videos["path"].duplicated().any():
        raise ValueError("Duplicate video paths in video manifest")
    if not set(videos["label"]).issubset({"real", "fake"}):
        raise ValueError("Labels must be exactly 'real' or 'fake'")
    if not set(videos["split"]).issubset(VALID_SPLITS):
        raise ValueError(f"Split values must be in {sorted(VALID_SPLITS)}")
    if videos[list(REQUIRED_COLUMNS)].eq("").any().any():
        raise ValueError("Required manifest fields cannot be empty")
    if frames_per_video < 1:
        raise ValueError("frames_per_video must be at least 1")
    if limit_videos is not None:
        if limit_videos < 1:
            raise ValueError("limit_videos must be at least 1")
        videos = videos.head(limit_videos).copy()

    frame_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for row in videos.to_dict(orient="records"):
        video_path = Path(row["path"]).expanduser()
        if not video_path.is_file():
            failures.append({"sample_id": row["sample_id"], "path": row["path"], "reason": "file_not_found"})
            continue
        destination_dir = output_root / row["dataset"] / row["split"] / row["label"]
        rows, error = extract_video(
            video_path, destination_dir, row["sample_id"], frames_per_video
        )
        for frame_row in rows:
            frame_row.update({
                "label": row["label"],
                "dataset": row["dataset"],
                "source_group": row["source_group"],
                "split": row["split"],
            })
        frame_rows.extend(rows)
        if error:
            failures.append({"sample_id": row["sample_id"], "path": row["path"], "reason": error})
        if not rows and not error:
            failures.append({"sample_id": row["sample_id"], "path": row["path"], "reason": "no_frames_extracted"})

    columns = [
        "frame_id", "frame_path", "video_sample_id", "video_path", "label",
        "dataset", "source_group", "split", "frame_index", "timestamp_sec",
    ]
    frame_manifest = pd.DataFrame(frame_rows, columns=columns)
    failure_manifest = pd.DataFrame(failures, columns=["sample_id", "path", "reason"])
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frame_manifest.to_csv(output_manifest, index=False)
    failure_path = output_manifest.with_name(f"{output_manifest.stem}.failures.csv")
    failure_manifest.to_csv(failure_path, index=False)
    return frame_manifest, failure_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Local video split CSV")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for extracted frames")
    parser.add_argument("--output-manifest", type=Path, required=True, help="Frame-level CSV path")
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--limit-videos", type=int, default=None, help="Process only the first N manifest rows")
    args = parser.parse_args()

    frames, failures = extract_manifest(
        args.manifest, args.output_root, args.output_manifest,
        args.frames_per_video, args.limit_videos,
    )
    print(f"Videos selected: {args.limit_videos or 'all'}")
    print(f"Frames extracted: {len(frames)}")
    print(f"Videos with extraction issues: {len(failures)}")
    if not frames.empty:
        print("\nFrame counts by split and label:")
        print(pd.crosstab(frames["split"], frames["label"]).to_string())
    print(f"Frame manifest: {args.output_manifest}")
    print(f"Failure report: {args.output_manifest.with_name(args.output_manifest.stem + '.failures.csv')}")


if __name__ == "__main__":
    main()
