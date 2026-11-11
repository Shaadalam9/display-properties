"""Propose each scene's event window from its geometry, for checking by eye.

    uv run python propose_events.py                     # placeholder windows -> data/review/events/
    uv run python propose_events.py --apply             # write the proposals into study/scenes.json
    uv run python propose_events.py --scene kobe_calm   # given scenes, even with a window set by hand

The event window (``event_s``, source-clip seconds) decides when the display appears in
"only around the event" (with ``lead_time_s`` before it) and when the explanation shows, so it
has to match what happens in the clip. The proposal, per scene:

* risky scenes: centred on the moment the nearest pedestrian ahead is closest (within 20 m),
  otherwise on the strongest braking;
* "Waiting" scenes: from the moment the car brakes towards its stop;
* calm scenes ("Keeping lane"): there is no real event, so the window sits in the middle of the
  clip; the display then explains routine driving.

The window starts at least ``lead`` seconds into the clip, so "only around the event" and
"always" differ before it. For every scene a sheet shows speed and pedestrian distance over the
clip with the proposed window, and frames at its start, middle and end. Check every sheet;
``--apply`` writes the proposals and sets ``event_checked`` to false until a person confirms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from avprops.clip_render import read_geometry
from avprops.common import PROJECT_ROOT, load_config, load_scenes, resolve

OUT = PROJECT_ROOT / "data" / "review" / "events"
PLACEHOLDER = [10.0, 15.0]  # the window the scanner gave every new scene


def tracks(geometry, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: time (s), speed (km/h, smoothed over 1 s), nearest pedestrian ahead (m)."""
    fps = geometry.fps
    phase = np.array([float((f.get("ribbon") or {}).get("phase_m") or 0.0) for f in geometry.frames])
    k = max(1, int(fps))
    speed = np.convolve(np.gradient(phase) * fps * 3.6, np.ones(k) / k, mode="same").clip(0, None)
    nearest = []
    for f in geometry.frames:
        ahead = [float(p["dist"]) for p in f.get("persons") or []
                 if p.get("dist") and 0.25 * width < (p["box"][0] + p["box"][2]) / 2 < 0.75 * width]
        nearest.append(min(ahead) if ahead else np.nan)
    return np.array([f["i"] / fps for f in geometry.frames]), speed, np.array(nearest)


def propose(scene: dict, t: np.ndarray, speed: np.ndarray, nearest: np.ndarray,
            length: float = 5.0, lead: float = 3.5) -> tuple[float, float, str]:
    start, end = scene["start_s"], scene["start_s"] + scene["duration_s"]
    inside = (t >= start + lead) & (t <= end - 1.0)
    how = (scene.get("how") or "").lower()
    if scene.get("criticality") == "high" and np.nanmin(np.where(inside, nearest, np.nan), initial=np.inf) < 20:
        centre = float(t[inside][np.nanargmin(nearest[inside])])
        reason = f"nearest pedestrian ahead {np.nanmin(nearest[inside]):.0f} m"
        first = centre - length / 2
    elif "wait" in how or scene.get("criticality") == "high":
        decel = -np.gradient(speed)
        idx = np.flatnonzero(inside)
        first = float(t[idx[np.argmax(decel[idx])]]) - 0.5
        reason = f"strongest braking (speed {speed[idx].max():.0f} -> {speed[idx].min():.0f} km/h)"
    else:
        first = (start + end) / 2 - length / 2
        reason = "routine driving: no event, middle of the clip"
    first = float(np.clip(first, start + lead, end - length))
    return round(first, 1), round(first + length, 1), reason


def sheet(scene: dict, clip: Path, t, speed, nearest, window, reason: str, out: Path) -> None:
    start, end = scene["start_s"], scene["start_s"] + scene["duration_s"]
    w, h = 960, 200
    plot = np.full((h, w, 3), 255, np.uint8)
    x = lambda s: int((s - start) / (end - start) * (w - 1))  # noqa: E731
    cv2.rectangle(plot, (x(window[0]), 0), (x(window[1]), h), (210, 235, 255), -1)
    inside = (t >= start) & (t <= end)
    for values, colour, top in ((speed, (200, 80, 0), max(30.0, np.nanmax(speed[inside]))),
                                (nearest, (0, 0, 200), 40.0)):
        pts = [(x(s), int(h - 10 - min(v, top) / top * (h - 30))) for s, v in zip(t[inside], values[inside])
               if not np.isnan(v)]
        if len(pts) > 1:
            cv2.polylines(plot, [np.array(pts, np.int32)], False, colour, 2, cv2.LINE_AA)
    cv2.putText(plot, f"{scene['id']}: event {window[0]}-{window[1]} s ({reason})", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(plot, "speed (km/h)", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 80, 0), 1, cv2.LINE_AA)
    cv2.putText(plot, "nearest pedestrian ahead (m, 0-40)", (130, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 200), 1, cv2.LINE_AA)
    cap = cv2.VideoCapture(str(clip))
    frames = []
    for s in (window[0] - 1.0, window[0], (window[0] + window[1]) / 2, window[1]):
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
        ok, img = cap.read()
        img = cv2.resize(img, (240, 135)) if ok else np.zeros((135, 240, 3), np.uint8)
        cv2.putText(img, f"{s:.1f} s", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(img)
    cap.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([plot, np.hstack(frames)]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", nargs="+", help="scene ids (default: every scene with a clip)")
    parser.add_argument("--apply", action="store_true", help="write the proposals into the scenes file")
    args = parser.parse_args()
    config = load_config()
    proposals = {}
    for scene in load_scenes(config):
        if not scene.get("clip") or (args.scene and scene["id"] not in args.scene):
            continue
        if not args.scene and scene["event_s"] != PLACEHOLDER:
            continue  # a window set by hand stays, unless the scene is named
        path = resolve(scene["geometry"]) if scene.get("geometry") else \
            PROJECT_ROOT / "data" / "geometry" / f"{scene['id']}_geometry.jsonl.gz"
        geometry = read_geometry(path)
        t, speed, nearest = tracks(geometry, geometry.size[0])
        window = propose(scene, t, speed, nearest)
        proposals[scene["id"]] = window
        sheet(scene, resolve(scene["clip"]), t, speed, nearest, window[:2], window[2], OUT / f"{scene['id']}.jpg")
        print(f"{scene['id']:22s} now {scene['event_s']}  proposed {list(window[:2])}  ({window[2]})")
    if args.apply:
        scenes_path = resolve(config["scenes"])
        data = json.loads(scenes_path.read_text())
        for scene in data["scenes"]:
            if scene["id"] in proposals:
                scene["event_s"] = list(proposals[scene["id"]][:2])
                scene["event_checked"] = False
        scenes_path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {len(proposals)} event windows to {scenes_path}")
    print(f"sheets in {OUT}")


if __name__ == "__main__":
    main()
