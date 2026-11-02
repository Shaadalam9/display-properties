"""Find calm and risky stretches in long dashcam videos, to choose stimulus scenes.

The video is decoded at a low frame rate (``fps``) and size by ffmpeg, road users are
detected with YOLO, and every frame gets a risk score from the road users ahead:

* a road user counts when its box is inside a corridor ahead of the car (the middle
  ``corridor`` share of the image width, below the image's upper third);
* its closeness is the box height as a share of the frame height (nearer = taller);
* pedestrians, cyclists and motorcyclists weigh ``vru_weight`` times as much as cars,
  buses and trucks;
* frame risk = the largest weighted closeness ahead.

A window of ``window_s`` seconds scores the 90th percentile of its frames' risk, so one
glitchy frame does not make a window risky. Windows that contain a cut (a large jump in
the colour histogram between samples), or lie in the first or last ``skip_s`` seconds
(intros, outros), are left out, and so are dark windows (dusk, night: the detector
misses road users there, which would make them look calm). This is a screen for review,
not a measure of danger: pick the final scenes by eye from the contact sheets.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

VRU = {0: "person", 1: "bicycle", 3: "motorcycle"}
VEHICLES = {2: "car", 5: "bus", 7: "truck"}
WIDTH = 640  # decode width; detection and the risk score work on this size


@dataclass
class ScanSettings:
    fps: float = 2.0
    window_s: float = 14.0
    step_s: float = 2.0
    skip_s: float = 60.0
    corridor: float = 0.4
    vru_weight: float = 1.5
    cut_threshold: float = 0.5   # histogram correlation below this between samples = a cut
    model: str = "yolo26s.pt"


def probe(path: Path) -> tuple[float, int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "format=duration:stream=width,height", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    stream = info["streams"][0]
    return float(info["format"]["duration"]), int(stream["width"]), int(stream["height"])


def frames(path: Path, fps: float) -> Iterator[np.ndarray]:
    """Frames at ``fps``, WIDTH pixels wide, decoded by ffmpeg (far faster than seeking)."""
    _, w, h = probe(path)
    height = int(round(h * WIDTH / w / 2) * 2)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={WIDTH}:{height}",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size = WIDTH * height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(height, WIDTH, 3)
    finally:
        proc.kill()
        proc.wait()


def frame_risk(boxes: np.ndarray, classes: np.ndarray, height: int, settings: ScanSettings) -> tuple[float, int, int]:
    """(risk, road users ahead, vulnerable road users ahead) of one frame."""
    lo, hi = WIDTH * (0.5 - settings.corridor / 2), WIDTH * (0.5 + settings.corridor / 2)
    risk, ahead, vru = 0.0, 0, 0
    for (x0, y0, x1, y1), cls in zip(boxes, classes):
        cx = (x0 + x1) / 2
        if not lo <= cx <= hi or y1 < height / 3:
            continue
        weight = settings.vru_weight if int(cls) in VRU else 1.0
        risk = max(risk, float(weight * (y1 - y0) / height))
        ahead += 1
        vru += int(cls) in VRU
    return risk, ahead, vru


def scan(path: Path, settings: ScanSettings, device: str | None = None) -> list[dict]:
    """Per-sample records: time, risk, road users ahead, VRUs ahead, histogram change."""
    from ultralytics import YOLO

    model = YOLO(settings.model)
    classes = sorted(VRU) + sorted(VEHICLES)
    records, last_hist = [], None
    for k, image in enumerate(frames(path, settings.fps)):
        hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        change = 1.0 if last_hist is None else float(cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL))
        last_hist = hist
        result = model.predict(image, classes=classes, conf=0.35, verbose=False, device=device)[0]
        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))
        cls = result.boxes.cls.cpu().numpy() if len(result.boxes) else np.zeros(0)
        risk, ahead, vru = frame_risk(boxes, cls, image.shape[0], settings)
        records.append({"t": round(k / settings.fps, 2), "risk": round(risk, 4), "ahead": ahead, "vru": vru,
                        "hist_corr": round(change, 4), "n": int(len(boxes))})
    return records


def windows(records: list[dict], duration: float, settings: ScanSettings) -> list[dict]:
    """Score every window of window_s seconds, every step_s seconds."""
    if not records:
        return []
    t = np.array([r["t"] for r in records])
    risk = np.array([r["risk"] for r in records])
    out = []
    for start in np.arange(settings.skip_s, duration - settings.skip_s - settings.window_s, settings.step_s):
        inside = (t >= start) & (t < start + settings.window_s)
        if inside.sum() < settings.window_s * settings.fps * 0.8:
            continue
        cut = any(records[i]["hist_corr"] < settings.cut_threshold for i in np.flatnonzero(inside)[1:])
        out.append({
            "start_s": round(float(start), 1), "end_s": round(float(start + settings.window_s), 1),
            "risk": round(float(np.percentile(risk[inside], 90)), 4),
            "mean_risk": round(float(np.mean(risk[inside])), 4),
            "vru_ahead": int(max(records[i]["vru"] for i in np.flatnonzero(inside))),
            "road_users": round(float(np.mean([records[i]["n"] for i in np.flatnonzero(inside)])), 1),
            "cut": cut,
        })
    return out


def brightness(path: Path, window: dict, n: int = 3) -> float:
    """Mean luma (0-255) of n frames across a window."""
    values = []
    for k in range(n):
        t = window["start_s"] + (k + 0.5) * (window["end_s"] - window["start_s"]) / n
        proc = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                               "-vf", "scale=160:-2,format=gray", "-f", "rawvideo", "-"], capture_output=True, check=True)
        values.append(float(np.frombuffer(proc.stdout, np.uint8).mean()))
    return float(np.mean(values))


def pick(scored: list[dict], n: int, low: bool, min_gap_s: float = 30.0, accept=None) -> list[dict]:
    """The n calmest (low=True) or riskiest windows without a cut, at least min_gap_s apart.
    Calm windows also need some traffic, so the display has something to show. ``accept``
    (window -> bool) can reject more, e.g. dark windows; it is only called on candidates."""
    pool = [w for w in scored if not w["cut"] and (not low or w["road_users"] >= 2)]
    # ties on the 90th percentile are broken by the mean, so the window that holds most of
    # a risky (or calm) stretch wins over one that only touches it
    pool.sort(key=lambda w: (w["risk"], w["mean_risk"]), reverse=not low)
    chosen: list[dict] = []
    for w in pool:
        if all(abs(w["start_s"] - c["start_s"]) >= min_gap_s for c in chosen) and (accept is None or accept(w)):
            chosen.append(w)
        if len(chosen) == n:
            break
    return chosen


def contact_sheet(path: Path, window: dict, out: Path, n: int = 5) -> None:
    """n frames across a window, side by side, for reviewing by eye."""
    tiles = []
    for k in range(n):
        t = window["start_s"] + (k + 0.5) * (window["end_s"] - window["start_s"]) / n
        proc = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                               "-vf", "scale=384:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
                              capture_output=True, check=True)
        tiles.append(cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 80])
