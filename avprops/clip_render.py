"""Render the study's video stimuli with every display property as a parameter.

The stimuli are real dashcam clips with an explanation overlay drawn on top. The
overlay is drawn here, from per-frame geometry, so each property of the display can
be set on its own and the low and high clip of a pair differ in that property only.

The geometry is one JSON line per frame, written by extract_geometry.py (or by any
other pipeline that produces the same fields):

    <tag>_geometry.jsonl.gz   line 0: header (clip_video, fps, width, height, camera
                              constants); then one line per frame: the planned-path
                              centreline, the pedestrians and vehicles (mask polygon,
                              box, distance, track id, score), the explanation ramp and
                              label text; extract_geometry.py adds "others" (detected
                              road users the vehicle does not react to) and "conf"
    <tag>_occlusion.mp4       grayscale map; 255 = something in front of the road

Rendering needs no GPU and no models. ``Style`` holds every parameter, with neutral
defaults; each property's low and high level in study/properties.json is a set of
overrides of that baseline.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _hex(code: str) -> tuple[int, int, int]:
    return tuple(int(code[i:i + 2], 16) for i in (1, 3, 5))


# The four palettes of the preference study this one feeds (palette_id 0-3), with their
# colours as RGB: "target" fills, outlines and marks road users, "trajectory" draws the
# path, "highlight" is the accent (here: the second road-user class when class_colours is
# on), "label_text"/"label_background" the text panels. "far" is where the path's
# uncertainty gradient ends; the palettes define none, so it is a red-ish tone per palette.
PALETTES = {
    0: {"name": "automotive_standard", "target": _hex("#FFDC00"), "trajectory": _hex("#5AC8F5"),
        "highlight": _hex("#FFA000"), "label_text": _hex("#FFFFFF"), "label_background": _hex("#000000"),
        "far": _hex("#FF4678")},
    1: {"name": "accessibility_okabe_ito", "target": _hex("#56B4E9"), "trajectory": _hex("#009E73"),
        "highlight": _hex("#E69F00"), "label_text": _hex("#111111"), "label_background": _hex("#F0E442"),
        "far": _hex("#D55E00")},
    2: {"name": "minimalist_hud", "target": _hex("#E0E0E0"), "trajectory": _hex("#48CAE4"),
        "highlight": _hex("#90E0EF"), "label_text": _hex("#FFFFFF"), "label_background": _hex("#222222"),
        "far": _hex("#FF6B6B")},
    3: {"name": "high_saliency", "target": _hex("#FF0055"), "trajectory": _hex("#CCFF00"),
        "highlight": _hex("#FF0055"), "label_text": _hex("#FFFFFF"), "label_background": _hex("#000000"),
        "far": _hex("#FF0000")},
}

HIGHLIGHT_STYLES = ("outline", "fill", "box")
SCOPES = ("relevant", "all")


@dataclass(frozen=True)
class Style:
    """Every renderable parameter. Defaults are the neutral baseline of the study.
    trajectory_alpha, mask_alpha, background_dim_alpha and palette_id have the ranges and
    defaults of the preference study's parameter space; the other defaults come from the
    literature review (e.g. marker size: the expert mean in Jansen et al., CHI 2025)."""

    # planned path
    trajectory_alpha: float = 0.55          # path opacity, 0-1; 0 = no path
    path_half_width_m: float = 0.6          # half of the band's width on the road
    path_length_m: float = 20.0             # how far ahead the path is drawn
    path_chevrons: bool = False             # direction arrows on the band
    path_motion: bool = False               # arrows flow forward
    path_uncertainty: float = 0.0           # 0 = one hue; 1 = full gradient to "far" with distance
    chevron_spacing_m: float = 4.0
    chevron_length_m: float = 1.6
    chevron_speed_mps: float = 5.0
    manoeuvre_cue: bool = False             # icon + text for the next manoeuvre
    # road users
    mask_alpha: float = 0.14                # road-user mask opacity, 0-0.7
    highlight_style: str = "fill"           # outline | fill | box
    highlight_scope: str = "relevant"       # relevant | all
    class_colours: bool = False             # pedestrians and vehicles in different hues
    marker_size: float = 0.15               # marker above pedestrians, x box height; 0 = none
    predicted_paths: bool = False           # where moving road users are heading (2 s)
    distance_labels: bool = False
    # explanation and vehicle state
    explanation_detail: int = 0             # 0 none, 1 icon, 2 why, 3 how + why
    confidence_indicator: int = 0           # 0 none, 1 label, 2 bar
    status_opacity: float = 0.0             # mode/speed panel; 0 = none
    panel_position: str = "top"             # top | bottom, for the explanation panel
    # timing
    show_when: str = "needed"               # needed (follows the explanation ramp) | always
    lead_time_s: float = 0.0                # overlay appears this much before the ramp
    fade_s: float = 0.5                     # 0 = instant on/off
    # appearance (fixed in the study, kept as parameters)
    background_dim_alpha: float = 0.06      # darkening of the rest of the scene, 0-0.4
    palette_id: int = 0                     # PALETTES key, 0-3
    contrast: float = 0.0                   # dark halo behind elements
    text_scale: float = 1.0

    @classmethod
    def from_overrides(cls, base: "Style", overrides: dict[str, Any]) -> "Style":
        unknown = set(overrides) - STYLE_PARAMETERS
        if unknown:
            raise ValueError(f"unknown style parameter(s): {', '.join(sorted(unknown))}")
        style = replace(base, **overrides)
        style.validate()
        return style

    def validate(self) -> None:
        if self.palette_id not in PALETTES:
            raise ValueError(f"unknown palette_id {self.palette_id!r}")
        for name, top in (("trajectory_alpha", 1.0), ("mask_alpha", 0.7), ("background_dim_alpha", 0.4)):
            if not 0.0 <= getattr(self, name) <= top:
                raise ValueError(f"{name} must be within 0-{top}")
        if self.highlight_style not in HIGHLIGHT_STYLES:
            raise ValueError(f"highlight_style must be one of {HIGHLIGHT_STYLES}")
        if self.highlight_scope not in SCOPES:
            raise ValueError(f"highlight_scope must be one of {SCOPES}")
        if self.show_when not in ("needed", "always") or self.panel_position not in ("top", "bottom"):
            raise ValueError("show_when must be needed|always and panel_position top|bottom")


STYLE_PARAMETERS = {f.name for f in fields(Style)}


def style_dict(style: Style) -> dict[str, Any]:
    return asdict(style)


# ---------------------------------------------------------------- geometry input

@dataclass
class Geometry:
    header: dict[str, Any]
    frames: list[dict[str, Any]]
    occlusion_path: Path | None

    @property
    def fps(self) -> float:
        return float(self.header.get("fps") or 30.0)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.header["width"]), int(self.header["height"])


def _scale_geometry(header: dict[str, Any], frames: list[dict[str, Any]], factor: float) -> None:
    """Scale every pixel coordinate in place (distances in metres stay as they are)."""
    header["width"] = int(round(header["width"] * factor / 2) * 2)
    header["height"] = int(round(header["height"] * factor / 2) * 2)
    for key in ("horizon_v", "vanish_u", "focal_px"):
        header["camera"][key] *= factor
    for frame in frames:
        ribbon = frame.get("ribbon") or {}
        if ribbon.get("centre"):
            ribbon["centre"] = (np.asarray(ribbon["centre"], float) * factor).tolist()
        for key in ("near_v", "far_v"):
            if ribbon.get(key) is not None:
                ribbon[key] *= factor
        for d in frame.get("persons", []) + frame.get("vehicles", []):
            d["box"] = [v * factor for v in d["box"]]
            if d.get("poly"):
                d["poly"] = (np.asarray(d["poly"], float) * factor).tolist()


def read_geometry(path: Path, max_width: int = 1280) -> Geometry:
    """Read a geometry dump. One made at a larger size (e.g. from 4K footage) is scaled
    down to max_width, so every stimulus is rendered at the same size."""
    opener = gzip.open if path.suffix == ".gz" else open
    header, frames = None, []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                if record.get("type") == "header":
                    header = record
                elif record.get("type") == "frame":
                    frames.append(record)
    if header is None or not frames:
        raise ValueError(f"{path}: no header or no frames (an interrupted extraction?)")
    frames.sort(key=lambda r: r["i"])
    if max_width and int(header["width"]) > max_width:
        _scale_geometry(header, frames, max_width / int(header["width"]))
    occlusion_path = None
    if header.get("occlusion_video"):
        named = Path(header["occlusion_video"])
        beside = path.parent / named.name  # the dump may have been copied elsewhere
        occlusion_path = named if named.exists() else (beside if beside.exists() else None)
    return Geometry(header, frames, occlusion_path)


# ---------------------------------------------------------------- camera model

class Camera:
    """Flat-ground pinhole model with the clip's calibrated constants."""

    def __init__(self, header: dict[str, Any]):
        cam = header["camera"]
        self.horizon = float(cam["horizon_v"])
        self.vanish = float(cam["vanish_u"])
        self.focal = float(cam["focal_px"])
        self.height = float(cam["cam_height_m"])

    def to_ground(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) below the horizon -> ground (lateral x, forward d) in metres."""
        d = self.focal * self.height / np.maximum(uv[:, 1] - self.horizon, 1e-3)
        return np.column_stack([(uv[:, 0] - self.vanish) * d / self.focal, d])

    def to_image(self, xd: np.ndarray) -> np.ndarray:
        d = np.maximum(xd[:, 1], 0.5)
        return np.column_stack([self.vanish + self.focal * xd[:, 0] / d, self.horizon + self.focal * self.height / d])


def _arc(ground: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ground, axis=0), axis=1))])


def path_on_ground(camera: Camera, ribbon: dict[str, Any] | None, length_m: float) -> np.ndarray | None:
    """The centreline on the ground, ordered near to far, cut at length_m of arc."""
    if not ribbon:
        return None
    centre = np.asarray(ribbon.get("centre") or [], dtype=float)
    if centre.ndim != 2 or len(centre) < 2:
        return None
    centre = centre[centre[:, 1] > camera.horizon + 1]
    if len(centre) < 2:
        return None
    if not ribbon.get("ordered"):
        centre = centre[np.argsort(-centre[:, 1])]  # row convention: nearest row first
    ground = camera.to_ground(centre)
    keep = _arc(ground) <= length_m
    return ground[keep] if keep.sum() >= 2 else None


def _normals(ground: np.ndarray) -> np.ndarray:
    tangent = np.gradient(ground, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
    return np.column_stack([tangent[:, 1], -tangent[:, 0]])


# ---------------------------------------------------------------- drawing helpers

def _font(size: float) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=max(8, int(round(size))))


def _mask(shape: tuple[int, int], draw) -> np.ndarray:
    """A uint8 mask (255 = covered) drawn by ``draw``."""
    m = np.zeros(shape, np.uint8)
    draw(m)
    return m


def _occlude(mask: np.ndarray, occlusion: np.ndarray | None) -> np.ndarray:
    """Hide the parts of a mask that something in front of the road covers."""
    if occlusion is None:
        return mask
    return (mask.astype(np.float32) * (1 - occlusion)).astype(np.uint8)


def _halo(mask: np.ndarray, contrast: float) -> np.ndarray:
    if contrast <= 0:
        return np.zeros_like(mask)
    k = 1 + 2 * max(1, int(round(4 * contrast)))
    return cv2.subtract(cv2.dilate(mask, np.ones((k, k), np.uint8)), mask)


class Overlay:
    """A colour + alpha layer the size of the frame, composited at the end. Each paint
    touches only the bounding box of its mask, which keeps a frame cheap to draw."""

    def __init__(self, width: int, height: int):
        self.colour = np.zeros((height, width, 3), np.float32)
        self.alpha = np.zeros((height, width), np.float32)

    def paint(self, mask: np.ndarray, rgb, opacity: float) -> None:
        if opacity <= 0:
            return
        x, y, w, h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            return
        roi = (slice(y, y + h), slice(x, x + w))
        a = (mask[roi].astype(np.float32) * (float(opacity) / 255.0))[..., None]
        rgb = np.asarray(rgb, np.float32)
        colour = rgb[roi] if rgb.ndim == 3 else rgb
        self.colour[roi] = self.colour[roi] * (1 - a) + colour * a
        self.alpha[roi] = self.alpha[roi] + a[..., 0] * (1 - self.alpha[roi])


# ---------------------------------------------------------------- renderer

class ClipRenderer:
    """Draws the display over a clip, frame by frame, for one style."""

    def __init__(self, geometry: Geometry, scene: dict[str, Any] | None = None):
        self.geometry = geometry
        self.scene = scene or {}
        # the scene's camera wins over the one stored with the geometry, so a calibration
        # can be corrected in scenes.json without extracting the geometry again
        self.camera = Camera({"camera": self.scene["camera"]} if self.scene.get("camera") else geometry.header)
        self.width, self.height = geometry.size
        self.fps = geometry.fps
        self._speed = self._speed_track()
        self._confidence = self._confidence_track()
        self._tracks = self._track_positions()

    # --- per-clip tracks

    def _speed_track(self) -> np.ndarray:
        """km/h per frame from the travelled-metres phase (``ribbon.phase_m``), when the geometry has it."""
        phases = [(f.get("ribbon") or {}).get("phase_m") for f in self.geometry.frames]
        if len(phases) < 2 or any(p is None for p in phases):
            return np.full(len(phases), np.nan)
        speed = np.clip(np.gradient(np.asarray(phases, float)) * self.fps * 3.6, 0, None)
        k = max(1, int(self.fps))
        return np.convolve(speed, np.ones(k) / k, mode="same")

    def _confidence_track(self) -> np.ndarray:
        """Detection confidence of the road users the vehicle reacts to, smoothed over 1 s."""
        values, last = [], 1.0
        for f in self.geometry.frames:
            scores = [float(d.get("conf", d.get("score", 1.0))) for d in self._relevant(f)]
            last = float(np.mean(scores)) if scores else last
            values.append(last)
        k = max(1, int(self.fps))
        padded = np.pad(np.asarray(values), (k, k), mode="edge")
        return np.convolve(padded, np.ones(k) / k, mode="same")[k:-k]

    def _track_positions(self) -> dict[int, dict[int, np.ndarray]]:
        """Ground position (x, d) of every tracked road user, per frame."""
        tracks: dict[int, dict[int, np.ndarray]] = {}
        for idx, f in enumerate(self.geometry.frames):
            for det in self._everyone(f):
                tid = int(det.get("tid", -1))
                if tid < 0:
                    continue
                x0, _, x1, y1 = det["box"]
                if y1 <= self.camera.horizon + 2:
                    continue
                tracks.setdefault(tid, {})[idx] = self.camera.to_ground(np.array([[(x0 + x1) / 2, y1]], float))[0]
        return tracks

    # --- which road users

    @staticmethod
    def _relevant(frame: dict) -> list[dict]:
        return ([dict(d, kind="person") for d in frame.get("persons") or []]
                + [dict(d, kind="vehicle") for d in frame.get("vehicles") or [] if d.get("relevant")])

    def _everyone(self, frame: dict) -> list[dict]:
        persons = [dict(d, kind="person") for d in frame.get("persons") or []]
        vehicles = [dict(d, kind="vehicle") for d in frame.get("vehicles") or []]
        others = [dict(d, kind=d.get("kind", "vehicle" if d.get("cls") not in ("person", "bicycle") else "person"))
                  for d in frame.get("others") or []]
        return persons + vehicles + others

    def _pedestrians(self, frame: dict) -> list[dict]:
        """People on foot: detected persons that are not riding a bicycle or motorcycle (a
        rider's box sits on top of the two-wheeler's box)."""
        everyone = self._everyone(frame)
        # two-wheeler class names of both geometry sources ("bike"/"moto" or COCO's names)
        wheels = [d["box"] for d in everyone if d.get("cls") in ("bicycle", "motorcycle", "bike", "moto")]

        def riding(box) -> bool:
            x0, y0, x1, y1 = box
            for wx0, wy0, wx1, wy1 in wheels:
                overlap = max(0, min(x1, wx1) - max(x0, wx0))
                if overlap > 0.5 * (x1 - x0) and y1 > wy0 and y1 < wy1 + 0.25 * (wy1 - wy0):
                    return True
            return False

        return [d for d in everyone if d["kind"] == "person" and d.get("cls", "person") == "person"
                and not riding(d["box"])]

    def _shown(self, frame: dict, style: Style) -> list[dict]:
        return self._relevant(frame) if style.highlight_scope == "relevant" else self._everyone(frame)

    # --- timing

    def visibility(self, style: Style) -> np.ndarray:
        """Overlay strength per frame (0..1) after show_when, lead time and fade. The event
        window is the scene's ``event_s`` when given, else the ramp stored in the geometry."""
        if self.scene.get("event_s"):
            t0, t1 = self.scene["event_s"]
            ramp = np.asarray([float(t0 <= f["i"] / self.fps <= t1) for f in self.geometry.frames])
        else:
            ramp = np.asarray([float(f.get("ramp", 1.0)) for f in self.geometry.frames])
        if style.show_when == "always":
            on = np.ones_like(ramp)
        else:
            on = (ramp > 0.05).astype(float)
            lead = int(round(style.lead_time_s * self.fps))
            if lead > 0:
                on = np.array([on[i:i + lead + 1].max() for i in range(len(on))])
        n = int(round(style.fade_s * self.fps))
        if n > 1 and len(on) > 0:
            on = np.convolve(np.pad(on, (n, n), mode="edge"), np.ones(n) / n, mode="same")[n:-n]
        return np.clip(on, 0, 1)

    # --- layers

    def _draw_path(self, ov: Overlay, frame: dict, style: Style, pal: dict, index: int, occlusion) -> None:
        if style.trajectory_alpha <= 0:
            return
        ground = path_on_ground(self.camera, frame.get("ribbon"), style.path_length_m)
        if ground is None:
            return
        shape = (self.height, self.width)
        normal = _normals(ground)
        left = self.camera.to_image(ground - normal * style.path_half_width_m)
        right = self.camera.to_image(ground + normal * style.path_half_width_m)
        band = _mask(shape, lambda m: cv2.fillPoly(
            m, [np.vstack([left, right[::-1]]).round().astype(np.int32)], 255, cv2.LINE_AA))
        band = _occlude(band, occlusion)
        ov.paint(_halo(band, style.contrast), (0, 0, 0), 0.6 * style.contrast)

        if style.path_uncertainty > 0:
            # colour along the path: near = path hue, far = "far" hue, by image row
            rows = np.arange(self.height, dtype=np.float32)
            near_v, far_v = left[0, 1], left[-1, 1]
            t = np.clip((near_v - rows) / max(near_v - far_v, 1.0), 0, 1) * style.path_uncertainty
            colour = (np.outer(1 - t, pal["trajectory"]) + np.outer(t, pal["far"]))[:, None, :]
            ov.paint(band, np.broadcast_to(colour, (self.height, self.width, 3)), style.trajectory_alpha)
        else:
            ov.paint(band, pal["trajectory"], style.trajectory_alpha)

        if style.path_chevrons:
            phase = 0.0
            if style.path_motion:
                phase = float((frame.get("ribbon") or {}).get("phase_m", 0.0)) + index / self.fps * style.chevron_speed_mps
            arc = _arc(ground)
            marks = []
            for s in np.arange(style.chevron_spacing_m - phase % style.chevron_spacing_m, arc[-1] - style.chevron_length_m,
                               style.chevron_spacing_m):
                at = lambda a, col: np.interp(a, arc, col)  # noqa: E731
                tip = np.array([at(s + style.chevron_length_m, ground[:, 0]), at(s + style.chevron_length_m, ground[:, 1])])
                base = np.array([at(s, ground[:, 0]), at(s, ground[:, 1])])
                side = np.array([at(s, normal[:, 0]), at(s, normal[:, 1])])
                w = style.path_half_width_m * 0.9
                pts = np.array([base - side * w, tip, base + side * w, base + (tip - base) * 0.45])
                marks.append(self.camera.to_image(pts).round().astype(np.int32))
            if marks:
                chev = _mask(shape, lambda m: cv2.fillPoly(m, marks, 255, cv2.LINE_AA))
                chev = _occlude(chev, occlusion)
                ov.paint(chev, (255, 255, 255), min(1.0, style.trajectory_alpha + 0.2))

    def _colour_for(self, det: dict, style: Style, pal: dict):
        return pal["highlight"] if style.class_colours and det["kind"] == "vehicle" else pal["target"]

    def _draw_highlights(self, ov: Overlay, frame: dict, style: Style, pal: dict) -> list[dict]:
        shape = (self.height, self.width)
        shown = self._shown(frame, style)
        thick = max(2, int(round(self.height / 300)))
        for det in shown:
            colour = self._colour_for(det, style, pal)
            x0, y0, x1, y1 = (int(v) for v in det["box"])
            poly = np.asarray(det.get("poly") or [], dtype=float)
            has_poly = poly.ndim == 2 and poly.shape[1:] == (2,) and len(poly) >= 3
            pts = poly.round().astype(np.int32) if has_poly else np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)
            if style.mask_alpha > 0:
                if style.highlight_style == "fill":
                    m = _mask(shape, lambda m: cv2.fillPoly(m, [pts], 255, cv2.LINE_AA))
                    opacity = style.mask_alpha
                elif style.highlight_style == "outline":
                    m = _mask(shape, lambda m: cv2.polylines(m, [pts], True, 255, thick, cv2.LINE_AA))
                    opacity = min(1.0, style.mask_alpha * 2)
                else:
                    m = _mask(shape, lambda m: cv2.rectangle(m, (x0, y0), (x1, y1), 255, thick, cv2.LINE_AA))
                    opacity = min(1.0, style.mask_alpha * 2)
                ov.paint(_halo(m, style.contrast), (0, 0, 0), 0.6 * style.contrast)
                ov.paint(m, colour, opacity)
        if style.marker_size > 0:
            # markers go on every pedestrian, whichever road users are highlighted
            for det in self._pedestrians(frame):
                x0, y0, x1, y1 = (int(v) for v in det["box"])
                # marker_size is read as the marker's radius in metres, so it shrinks with
                # distance like a real object; closer than 5 m it stops growing
                px_per_m = (self.camera.focal / max(5.0, float(det["dist"])) if det.get("dist")
                            else (y1 - y0) / 1.7)
                r = max(2, int(round(style.marker_size * px_per_m)))
                centre = ((x0 + x1) // 2, y0 - r - 4)
                m = _mask(shape, lambda m: cv2.circle(m, centre, r, 255, -1, cv2.LINE_AA))
                ov.paint(_halo(m, max(style.contrast, 0.3)), (0, 0, 0), 0.5)
                ov.paint(m, self._colour_for(det, style, pal), 0.95)
        return shown

    def _draw_predictions(self, ov: Overlay, frame_index: int, style: Style, pal: dict, shown: list[dict]) -> None:
        if not style.predicted_paths:
            return
        k = max(1, int(round(1.0 * self.fps)))  # velocity over the last second; shorter windows jitter
        lines = []
        for det in shown:
            track = self._tracks.get(int(det.get("tid", -1)), {})
            now, before = track.get(frame_index), track.get(frame_index - k)
            if now is None or before is None:
                continue
            # Only sideways movement: the camera car's own forward speed is unknown, and it
            # makes everything ahead seem to approach; lateral ground position is unaffected.
            velocity = np.array([(now[0] - before[0]) / (k / self.fps), 0.0])
            if abs(velocity[0]) < 0.4:
                continue
            ahead = np.array([now + velocity * t for t in np.linspace(0, 2.0, 12)])
            ahead = ahead[ahead[:, 1] > 1.0]
            if len(ahead) >= 2:
                lines.append(self.camera.to_image(ahead).round().astype(np.int32))
        if lines:
            thick = max(2, int(round(self.height / 240)))

            def draw(m):
                for p in lines:
                    cv2.polylines(m, [p], False, 255, thick, cv2.LINE_AA)
                    cv2.circle(m, (int(p[-1][0]), int(p[-1][1])), thick * 2, 255, -1, cv2.LINE_AA)

            ov.paint(_mask((self.height, self.width), draw), (255, 255, 255), 0.85)

    def _dim_mask(self, frame: dict, style: Style) -> np.ndarray:
        keep = np.zeros((self.height, self.width), np.uint8)
        for det in self._shown(frame, style):
            x0, y0, x1, y1 = (int(v) for v in det["box"])
            cv2.rectangle(keep, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), 255, -1)
        ground = path_on_ground(self.camera, frame.get("ribbon"), style.path_length_m) if style.trajectory_alpha > 0 else None
        if ground is not None:
            n = _normals(ground)
            w = style.path_half_width_m + 0.5
            pts = np.vstack([self.camera.to_image(ground - n * w), self.camera.to_image(ground + n * w)[::-1]])
            cv2.fillPoly(keep, [pts.round().astype(np.int32)], 255)
        return 1.0 - cv2.GaussianBlur(keep, (0, 0), 12).astype(np.float32) / 255.0

    def _draw_text(self, image: Image.Image, frame: dict, style: Style, pal: dict, shown, index: int,
                   strength: float) -> None:
        draw = ImageDraw.Draw(image, "RGBA")
        unit = self.height / 720 * style.text_scale
        a = int(round(255 * strength))

        def boxed(xy, text, size, anchor="la", panel_alpha=170):
            font = _font(size)
            box = draw.textbbox(xy, text, font=font, anchor=anchor)
            pad = int(size * 0.4)
            draw.rounded_rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), radius=pad,
                                   fill=pal["label_background"] + (int(round(panel_alpha * strength)),))
            draw.text(xy, text, font=font, fill=pal["label_text"] + (a,), anchor=anchor)
            return box

        if style.distance_labels:
            for det in shown:
                if det.get("dist") is not None:
                    x0, y0, x1, _ = det["box"]
                    boxed(((x0 + x1) / 2, y0 - 14 * unit), f"{float(det['dist']):.0f} m", 18 * unit, "md")

        panel_y = 36 * unit if style.panel_position == "top" else self.height - 110 * unit
        if style.explanation_detail >= 1:
            why, how = self.scene.get("why", ""), self.scene.get("how", "")
            text = {2: why, 3: f"{how}: {why[:1].lower() + why[1:]}" if how and why else why}.get(
                style.explanation_detail, "")
            icon = 34 * unit
            cx = self.width / 2 - (draw.textlength(text, font=_font(28 * unit)) / 2 + icon * 0.8 if text else 0)
            cy = panel_y + icon / 2
            # warning icon: a rounded triangle with an exclamation mark
            tri = [(cx, cy - icon / 2), (cx + icon / 2, cy + icon / 2), (cx - icon / 2, cy + icon / 2)]
            draw.polygon(tri, fill=pal["highlight"] + (a,))
            draw.text((cx, cy + icon * 0.12), "!", font=_font(icon * 0.6), fill=(0, 0, 0, a), anchor="mm")
            if text:
                boxed((cx + icon * 0.9, cy), text, 28 * unit, "lm")

        if style.manoeuvre_cue and self.scene.get("manoeuvre"):
            text = self.scene["manoeuvre"]
            x, y = 40 * unit, panel_y + (60 * unit if style.explanation_detail else 0)
            boxed((x + 64 * unit, y + 22 * unit), text, 24 * unit, "lm")
            d = -1 if "left" in text.lower() else 1 if "right" in text.lower() else 0
            cx, cy, s = x + 24 * unit, y + 22 * unit, 18 * unit
            if "stop" in text.lower() or "wait" in text.lower():
                draw.regular_polygon((cx, cy, s), 8, rotation=22.5, fill=(220, 40, 40, a))
            elif d == 0:
                draw.polygon([(cx, cy - s), (cx + s * .7, cy), (cx + s * .25, cy), (cx + s * .25, cy + s),
                              (cx - s * .25, cy + s), (cx - s * .25, cy), (cx - s * .7, cy)], fill=pal["trajectory"] + (a,))
            else:
                draw.polygon([(cx + d * s, cy), (cx, cy - s * .7), (cx, cy - s * .25), (cx - d * s, cy - s * .25),
                              (cx - d * s, cy + s * .25), (cx, cy + s * .25), (cx, cy + s * .7)], fill=pal["trajectory"] + (a,))

        if style.status_opacity > 0:
            speed = self._speed[index] if index < len(self._speed) else np.nan
            if np.isnan(speed) and self.scene.get("speed_kmh") is not None:
                speed = float(self.scene["speed_kmh"])
            text = "AUTOMATED" + ("" if np.isnan(speed) else f"   {speed:.0f} km/h")
            boxed((40 * unit, self.height - 36 * unit), text, 22 * unit, "ld", int(255 * style.status_opacity))

        if style.confidence_indicator:
            c = float(self._confidence[index]) if index < len(self._confidence) else 1.0
            x1, y1 = self.width - 40 * unit, self.height - 36 * unit
            if style.confidence_indicator == 1:
                boxed((x1, y1), "Confident" if c >= 0.6 else "Uncertain", 22 * unit, "rd")
            else:
                w, h = 180 * unit, 14 * unit
                boxed((x1, y1 - h - 10 * unit), "Confidence", 18 * unit, "rd")
                draw.rounded_rectangle((x1 - w, y1 - h, x1, y1), radius=h / 2, fill=(255, 255, 255, int(a * 0.35)))
                colour = pal["trajectory"] if c >= 0.6 else pal["far"]
                draw.rounded_rectangle((x1 - w, y1 - h, x1 - w + max(h, w * c), y1), radius=h / 2, fill=colour + (a,))

    # --- frame

    def render_frame(self, image_bgr: np.ndarray, index: int, style: Style, strength: float,
                     occlusion: np.ndarray | None = None) -> np.ndarray:
        if strength <= 0:
            return image_bgr
        frame = self.geometry.frames[index]
        pal = PALETTES[style.palette_id]
        out = image_bgr.astype(np.float32)
        if style.background_dim_alpha > 0:
            out *= 1 - (self._dim_mask(frame, style) * style.background_dim_alpha * strength)[..., None]
        ov = Overlay(self.width, self.height)
        self._draw_path(ov, frame, style, pal, index, occlusion)
        shown = self._draw_highlights(ov, frame, style, pal)
        self._draw_predictions(ov, index, style, pal, shown)
        a = (ov.alpha * strength)[..., None]
        out = out * (1 - a) + ov.colour[..., ::-1] * a
        out = np.clip(out, 0, 255).astype(np.uint8)
        if (style.distance_labels or style.explanation_detail or style.manoeuvre_cue or style.status_opacity > 0
                or style.confidence_indicator):
            image = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            self._draw_text(image, frame, style, pal, shown, index, strength)
            out = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return out


# ---------------------------------------------------------------- clips

def read_frames(path: Path, start: int, count: int, gray: bool = False,
                size: tuple[int, int] | None = None) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        for _ in range(count):
            ok, image = cap.read()
            if not ok:
                return
            if size and (image.shape[1], image.shape[0]) != size:
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            yield cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 if gray else image
    finally:
        cap.release()


def render_variants(geometry: Geometry, clip_video: Path, styles: list[tuple[str, Style | None]], out_dir: Path,
                    start_s: float = 0.0, duration_s: float | None = None, scene: dict | None = None) -> int:
    """Render several styles of one clip in a single pass over its frames, each to
    out_dir/<name>.mp4 as H.264 (browser playable). A style of None writes the clip with
    no display at all. Returns the frames per clip."""
    renderer = ClipRenderer(geometry, scene)
    start = int(round(start_s * geometry.fps))
    count = len(geometry.frames) - start if duration_s is None else int(round(duration_s * geometry.fps))
    count = max(0, min(count, len(geometry.frames) - start))
    strengths = {name: renderer.visibility(style) if style is not None else None for name, style in styles}
    occlusions = (read_frames(geometry.occlusion_path, start, count, gray=True, size=geometry.size)
                  if geometry.occlusion_path else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        writers = {name: cv2.VideoWriter(str(Path(tmp) / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         geometry.fps, geometry.size) for name, _ in styles}
        written = 0
        try:
            for k, image in enumerate(read_frames(clip_video, start, count, size=geometry.size)):
                occlusion = next(occlusions, None) if occlusions is not None else None
                i = start + k
                for name, style in styles:
                    if style is None:
                        writers[name].write(image)
                    else:
                        writers[name].write(renderer.render_frame(image, i, style, float(strengths[name][i]), occlusion))
                written += 1
        finally:
            for writer in writers.values():
                writer.release()
        for name, _ in styles:
            encode_h264(Path(tmp) / f"{name}.mp4", out_dir / f"{name}.mp4")
    return written


def render_clip(geometry: Geometry, clip_video: Path, style: Style, out_path: Path,
                start_s: float = 0.0, duration_s: float | None = None, scene: dict | None = None) -> int:
    """Render one stimulus clip to out_path (H.264)."""
    return render_variants(geometry, clip_video, [(out_path.stem, style)], out_path.parent, start_s, duration_s, scene)


def encode_h264(src: Path, dst: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to encode H.264, which browsers require")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", "-movflags", "+faststart", "-an", str(dst)], check=True)
