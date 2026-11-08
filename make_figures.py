"""Figures for the README, made from the rendered clips (public/stimuli/).

    uv run python make_figures.py          # -> docs/figures/

Writes:
    display.gif      the default display in one scene
    preview.gif      a ranking-screen preview: one property at its two levels
    scenes.jpg       a frame of every scene, during its event
    properties/<id>.gif  every property at its two levels side by side, labelled (from its
                     ranking preview)

The frames are small and low quality on purpose: they illustrate the stimuli, they are not
the stimuli (the footage is third-party dashcam video, see the README).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from avprops.common import PROJECT_ROOT, load_config, load_scenes

STIMULI = PROJECT_ROOT / "public" / "stimuli"
OUT = PROJECT_ROOT / "docs" / "figures"
TILE = (384, 216)
COUNTRY = {"US": "USA", "EG": "Egypt", "NL": "Netherlands", "ES": "Spain", "MX": "Mexico", "IN": "India",
           "VN": "Vietnam", "JP": "Japan", "AU": "Australia"}


def frame(path: Path, t: float, size: tuple[int, int] = TILE) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"no frame at {t:.1f} s in {path}")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def label(image: np.ndarray, text: str, height: int = 22) -> np.ndarray:
    """The frame with a dark bar below it (not over it: the display draws text at the top)."""
    bar = np.zeros((height, image.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (5, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([image, bar])


def grid(tiles: list[np.ndarray], cols: int, gap: int = 6) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    rows = -(-len(tiles) // cols)
    sheet = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 255, np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = tile
    return sheet


def event_time(scene: dict, offset: float = 1.5) -> float:
    """Seconds into the rendered clip, ``offset`` after the event starts."""
    return scene["event_s"][0] - scene["start_s"] + offset


def scene_name(scene: dict) -> str:
    place = (scene.get("source") or {}).get("city") or COUNTRY.get(scene.get("country", ""), scene.get("country", ""))
    risk = "risky" if scene.get("criticality") == "high" else "calm"
    return f"{place}, {risk}"


def gif(src: Path, dst: Path, start: float, duration: float, width: int, fps: int = 10) -> None:
    graph = f"fps={fps},scale={width}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{duration:.2f}", "-i", str(src),
                    "-filter_complex", graph, "-loop", "0", str(dst)], check=True)


def property_gif(prop: dict, dst: Path, width: int = 480, fps: int = 6, seconds: float = 3.5) -> None:
    """The property's ranking preview (low | high, or the four palettes 2 x 2) as a GIF, with
    the level named under each tile."""
    src = STIMULI / "previews" / f"{prop['id']}.mp4"
    if prop.get("variants"):
        names = ["low: standard"] + [f"high: {v.replace('_', ' ')}" for v in prop["variants"]]
    else:
        names = [f"low: {prop['low']}", f"high: {prop['high']}"]
    cap = cv2.VideoCapture(str(src))
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / fps))
    frames, k = [], 0
    while len(frames) < seconds * fps:
        ok, image = cap.read()
        if not ok:
            break
        if k % step == 0:
            frames.append(image)
        k += 1
    cap.release()
    h, w = frames[0].shape[:2]
    cols = 2
    rows = len(names) // cols
    tile_w, tile_h = width // cols, round(width / w * h) // rows
    out = []
    for image in frames:
        image = cv2.resize(image, (tile_w * cols, tile_h * rows), interpolation=cv2.INTER_AREA)
        tiles = [label(image[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w], names[r * cols + c])
                 for r in range(rows) for c in range(cols)]
        out.append(grid(tiles, cols, gap=4))
    tmp = dst.with_suffix(".mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out[0].shape[1], out[0].shape[0]))
    for image in out:
        writer.write(image)
    writer.release()
    gif(tmp, dst, 0, seconds, out[0].shape[1], fps=fps)
    tmp.unlink()


def save(image: np.ndarray, name: str) -> None:
    cv2.imwrite(str(OUT / name), image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(OUT / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    design = json.loads((PROJECT_ROOT / "public" / "design.json").read_text())
    scenes = {s["id"]: s for s in load_scenes(load_config())}  # with event times and sources
    rendered = [scenes[s["id"]] for s in design["scenes"] if (STIMULI / s["id"] / "baseline.mp4").exists()]

    # an animated example of the display, and of a ranking-screen preview
    show = scenes["valencia_risky"]
    gif(STIMULI / show["id"] / "baseline.mp4", OUT / "display.gif", event_time(show, -0.5), 4.0, 640)
    print(OUT / "display.gif")
    gif(STIMULI / "previews" / "mask_alpha.mp4", OUT / "preview.gif", 0.5, 3.0, 640, fps=8)
    print(OUT / "preview.gif")

    # every scene during its event
    save(grid([label(frame(STIMULI / s["id"] / "baseline.mp4", event_time(s)), scene_name(s)) for s in rendered], 4),
         "scenes.jpg")

    # every property as a labelled GIF
    (OUT / "properties").mkdir(exist_ok=True)
    for prop in design["properties"]:
        property_gif(prop, OUT / "properties" / f"{prop['id']}.gif", width=560 if prop.get("variants") else 480)
    print(OUT / "properties", "(17 GIFs)")


if __name__ == "__main__":
    main()
