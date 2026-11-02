"""Make overlay geometry for a source clip with a detector and tracker.

Writes the geometry format the clip renderer reads (see clip_render.py), so stimuli can
be made from any dashcam clip. It is a simple model of the scene:

* road users: YOLO segmentation with ByteTrack ids; distance from the flat-ground
  camera model at the bottom of the box (no depth model);
* which road users the car reacts to: those inside a corridor ahead of the car
  (``persons`` and ``vehicles`` with ``relevant``), the rest go to ``others``;
* planned path: a straight path in the car's lane (no lane detection or visual
  odometry), so a turn needs a hand-set ``path_curve`` or geometry from a lane-following pipeline;
* explanation ramp: 1 during the scene's ``event_s``, else 0.

Geometry from a pipeline that follows lanes and turns can be used instead: point the
scene's ``geometry`` field at it.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from avprops.clip_render import Camera, encode_h264

PERSON_CLASSES = {0: "person", 1: "bicycle", 3: "motorcycle"}
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}
SIZE = (1280, 720)  # geometry is made at this size; the camera constants refer to it
# used for a scene without its own constants (one whose calibration came with pipeline geometry)
DEFAULT_CAMERA = {"horizon_v": 448.0, "vanish_u": 636.0, "focal_px": 1000.0, "cam_height_m": 1.3}


def _path_centre(camera: Camera, offset_m: float, curve: float) -> list[list[float]]:
    """A path on the ground from 3 m to 45 m ahead, lateral = offset + curve * d^2."""
    d = np.arange(3.0, 45.0, 0.5)
    ground = np.column_stack([offset_m + curve * d ** 2, d])
    uv = camera.to_image(ground)
    keep = (uv[:, 0] > -200) & (uv[:, 0] < SIZE[0] + 200) & (uv[:, 1] < SIZE[1] + 50)
    return np.round(uv[keep], 2).tolist()


def extract(scene: dict[str, Any], clip: Path, out_dir: Path, model_name: str = "yolo26m-seg.pt",
            device: str | None = None, corridor_m: float = 3.5, reach_m: float = 35.0) -> Path:
    from ultralytics import YOLO  # heavy; only needed here

    camera_constants = scene.get("camera") or DEFAULT_CAMERA
    camera = Camera({"camera": camera_constants})
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    event = scene.get("event_s")
    path = _path_centre(camera, float(scene.get("path_offset_m", 0.0)), float(scene.get("path_curve", 0.0)))

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = scene["id"]
    geometry_path = out_dir / f"{tag}_geometry.jsonl.gz"
    occlusion_path = out_dir / f"{tag}_occlusion.mp4"
    occlusion_tmp = out_dir / f"{tag}_occlusion_mp4v.mp4"
    model = YOLO(model_name)
    classes = sorted(PERSON_CLASSES) + sorted(VEHICLE_CLASSES)

    writer = cv2.VideoWriter(str(occlusion_tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, SIZE)
    with gzip.open(geometry_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "type": "header", "version": 2, "source": "extract_geometry.py", "model": model_name,
            "clip_video": str(clip.resolve()), "fps": fps, "width": SIZE[0], "height": SIZE[1],
            "frame_count": total, "camera": camera_constants, "occlusion_video": str(occlusion_path.resolve()),
        }) + "\n")
        index = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            image = cv2.resize(image, SIZE, interpolation=cv2.INTER_AREA)
            result = model.track(image, persist=True, tracker="bytetrack.yaml", classes=classes, conf=0.3,
                                 verbose=False, device=device)[0]
            persons, vehicles, others = [], [], []
            occlusion = np.zeros((SIZE[1], SIZE[0]), np.uint8)
            boxes = result.boxes
            polys = result.masks.xy if result.masks is not None else [None] * len(boxes)
            for k in range(len(boxes)):
                cls = int(boxes.cls[k])
                x0, y0, x1, y1 = (int(round(v)) for v in boxes.xyxy[k].tolist())
                poly = polys[k]
                if poly is not None and len(poly) >= 3:
                    poly = cv2.approxPolyDP(poly.astype(np.float32), 1.5, True).reshape(-1, 2)
                    cv2.fillPoly(occlusion, [poly.round().astype(np.int32)], 255)
                    poly = np.round(poly, 1).tolist()
                else:
                    poly = None
                    cv2.rectangle(occlusion, (x0, y0), (x1, y1), 255, -1)
                dist, relevant = None, False
                if y1 > camera.horizon + 2:
                    x, d = camera.to_ground(np.array([[(x0 + x1) / 2, y1]], float))[0]
                    dist = round(float(np.hypot(x, d)), 2)
                    relevant = bool(abs(x) <= corridor_m and d <= reach_m)
                record = {"poly": poly, "box": [x0, y0, x1, y1], "score": round(float(boxes.conf[k]), 4),
                          "conf": round(float(boxes.conf[k]), 4), "dist": dist,
                          "tid": int(boxes.id[k]) if boxes.id is not None else -1,
                          "cls": PERSON_CLASSES.get(cls) or VEHICLE_CLASSES.get(cls)}
                if cls in PERSON_CLASSES:
                    (persons if relevant else others).append(record)
                else:
                    vehicles.append(dict(record, relevant=relevant))
            t = index / fps
            ramp = 1.0 if not event else float(event[0] <= t <= event[1])
            fh.write(json.dumps({"type": "frame", "i": index, "ramp": ramp, "label": "",
                                 "ribbon": {"centre": path, "near_v": path[0][1], "far_v": path[-1][1], "ordered": True},
                                 "persons": persons, "vehicles": vehicles, "others": others}) + "\n")
            soft = cv2.GaussianBlur(occlusion, (0, 0), 2)
            writer.write(cv2.cvtColor(soft, cv2.COLOR_GRAY2BGR))
            index += 1
    cap.release()
    writer.release()
    encode_h264(occlusion_tmp, occlusion_path)
    occlusion_tmp.unlink(missing_ok=True)
    return geometry_path
