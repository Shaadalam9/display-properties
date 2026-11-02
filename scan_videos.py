"""Propose calm and risky scenes from long dashcam videos.

    uv run --extra extract python scan_videos.py VIDEO.mp4 [VIDEO.mp4 ...] --out data/scan --device cuda:0

For each video it writes to the output folder:
    <video>_samples.jsonl   risk, road users and histogram change per sample
    <video>_windows.csv     every window with its score (90th percentile of frame risk)
    <video>_low_<k>.jpg     contact sheets of the calmest windows (with some traffic)
    <video>_high_<k>.jpg    contact sheets of the riskiest windows
and candidates.csv over all videos. Review the contact sheets, then copy the chosen
windows into study/scenes.json (clip, start_s, criticality, country). See avprops/scan.py
for how the risk is scored; it is a screen for review, not a measure of danger.

With ``--segments FILE`` (CSV with columns video, start_s, end_s), only windows inside a
listed stretch of their video are considered: for dataset videos, the stretches the
dataset's mapping lists as driving in the locality, which leaves out intros, walking
parts and other footage of the same upload.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from avprops.scan import ScanSettings, brightness, contact_sheet, pick, probe, scan, windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/scan"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--per-video", type=int, default=3, help="candidates of each kind per video")
    parser.add_argument("--model", default=ScanSettings.model)
    parser.add_argument("--segments", type=Path, help="CSV video,start_s,end_s: consider only these stretches")
    parser.add_argument("--min-brightness", type=float, default=70.0,
                        help="mean luma (0-255) below which a window counts as dark and is skipped")
    args = parser.parse_args()
    settings = ScanSettings(model=args.model)

    args.out.mkdir(parents=True, exist_ok=True)
    stretches: dict[str, list[tuple[float, float]]] = {}
    if args.segments:
        with args.segments.open(newline="") as fh:
            for row in csv.DictReader(fh):
                stretches.setdefault(row["video"], []).append((float(row["start_s"]), float(row["end_s"])))
    candidates = []
    for video in args.videos:
        stem = video.stem
        duration, _, _ = probe(video)
        samples_path = args.out / f"{stem}_samples.jsonl"
        if samples_path.exists():  # resumable: a scanned video is not scanned again
            records = [json.loads(line) for line in samples_path.read_text().splitlines() if line]
        else:
            records = scan(video, settings, args.device)
            samples_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        scored = windows(records, duration, settings)
        if args.segments:
            scored = [w for w in scored
                      if any(a <= w["start_s"] and w["end_s"] <= b for a, b in stretches.get(stem, []))]
        with (args.out / f"{stem}_windows.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["start_s", "end_s", "risk", "mean_risk", "vru_ahead", "road_users", "cut"])
            writer.writeheader()
            writer.writerows(scored)
        for kind, low in (("low", True), ("high", False)):
            bright = lambda w: brightness(video, w) >= args.min_brightness  # noqa: E731
            for k, w in enumerate(pick(scored, args.per_video, low, accept=bright)):
                sheet = args.out / f"{stem}_{kind}_{k}.jpg"
                contact_sheet(video, w, sheet)
                candidates.append({"video": stem, "kind": kind, "rank": k, **w,
                                   "brightness": round(brightness(video, w), 1), "sheet": sheet.name})
        print(f"{stem}: {duration / 60:.0f} min, {len(scored)} windows scored", flush=True)
    with (args.out / "candidates.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(candidates[0]) if candidates else ["video"])
        writer.writeheader()
        writer.writerows(candidates)


if __name__ == "__main__":
    main()
