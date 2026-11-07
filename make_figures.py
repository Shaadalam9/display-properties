"""Figures for the README, made from the rendered clips (public/stimuli/).

    uv run python make_figures.py          # -> docs/figures/

Writes:
    display.gif      the default display in one scene
    preview.gif      a ranking-screen preview: one property at its two levels
    scenes.jpg       a frame of every scene, during its event
    properties.jpg   every screened-in property at its low and high level
    palettes.jpg     the four colour palettes

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


def caption(width: int, text: str, height: int = 26) -> np.ndarray:
    bar = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(bar, text, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return bar


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

    # every property, low | high, in the scene where its two clips differ most
    checked = json.loads((PROJECT_ROOT / "data" / "stimulus_check.json").read_text())
    pairs = []
    for prop in design["properties"]:
        allowed = design["ab"]["scenes"][prop["id"]]
        scene = scenes[max(allowed, key=lambda s: checked.get(s, {}).get(prop["id"], {}).get("changed", 0))]
        # "when shown" differs before the event (and its lead time): the display is off or already on
        t = event_time(scene, -4.5) if prop["id"] == "show_when" else event_time(scene)
        high = f"{prop['id']}_high_{prop['variants'][0]}" if prop.get("variants") else f"{prop['id']}_high"
        low_img = label(frame(STIMULI / scene["id"] / f"{prop['id']}_low.mp4", t), f"low: {prop['low']}")
        high_text = f"high: {prop['variants'][0].replace('_', '-')} palette (one of three)" if prop.get("variants") \
            else f"high: {prop['high']}"
        high_img = label(frame(STIMULI / scene["id"] / f"{high}.mp4", t), high_text)
        pair = np.hstack([low_img, np.full((low_img.shape[0], 4, 3), 255, np.uint8), high_img])
        pairs.append(np.vstack([caption(pair.shape[1], f"{prop['name']}  ({prop['id']}, {scene_name(scene)})"), pair]))
    save(grid(pairs, 2, gap=10), "properties.jpg")

    # the four palettes in one scene
    scene = scenes["jaipur_risky"]
    t = event_time(scene)
    names = [("palette_id_low", "0 automotive standard"), ("palette_id_high_okabe_ito", "1 Okabe-Ito (accessible)"),
             ("palette_id_high_minimalist_hud", "2 minimalist HUD"), ("palette_id_high_high_saliency", "3 high saliency")]
    save(grid([label(frame(STIMULI / scene["id"] / f"{f}.mp4", t), text) for f, text in names], 4), "palettes.jpg")


if __name__ == "__main__":
    main()
