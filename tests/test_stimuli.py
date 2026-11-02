"""Every property shown to participants must be editable in a clip: its low and high
versions, rendered from the same geometry, must differ on screen."""

import numpy as np
import pytest

from avprops.clip_render import Camera, ClipRenderer, Geometry, Style
from avprops.common import load_config, load_properties
from render_stimuli import baseline_style, variants

W, H, FPS, N = 320, 180, 30.0, 60
CAMERA = {"horizon_v": 112.0, "vanish_u": 160.0, "focal_px": 250.0, "cam_height_m": 1.3}
SCENE = {"why": "Pedestrians crossing", "how": "Waiting", "manoeuvre": "Turning left"}


def synthetic_geometry() -> Geometry:
    cam = Camera({"camera": CAMERA})
    centre = cam.to_image(np.column_stack([np.zeros(80), np.linspace(3, 42, 80)])).round(2).tolist()
    frames = []
    for i in range(N):
        px = 100 + i  # a pedestrian walking right, tracked
        frames.append({
            "type": "frame", "i": i, "ramp": 1.0 if 25 <= i <= 50 else 0.0, "label": "",
            "ribbon": {"centre": centre, "ordered": True, "phase_m": i * 0.1},
            "persons": [{"poly": [[px, 120], [px + 10, 120], [px + 10, 150], [px, 150]], "box": [px, 120, px + 10, 150],
                         "dist": 9.0, "tid": 1, "score": 0.9 - i * 0.008, "conf": 0.9 - i * 0.008}],
            "vehicles": [{"poly": None, "box": [220, 115, 270, 145], "dist": 14.0, "tid": 2, "score": 0.8,
                          "relevant": False, "cls": "car"}],
            "others": [{"poly": None, "box": [20, 118, 28, 140], "dist": 20.0, "tid": 3, "score": 0.7, "cls": "person"}],
        })
    header = {"type": "header", "fps": FPS, "width": W, "height": H, "camera": CAMERA}
    return Geometry(header, frames, None)


def background() -> np.ndarray:
    image = np.full((H, W, 3), 90, np.uint8)
    image[:112] = (200, 170, 140)
    return image


def render_all(renderer: ClipRenderer, style: Style) -> np.ndarray:
    strength = renderer.visibility(style)
    image = background()
    return np.stack([renderer.render_frame(image, i, style, float(strength[i])) for i in range(N)])


@pytest.fixture(scope="module")
def setup():
    config = load_config()
    return config, ClipRenderer(synthetic_geometry(), SCENE), baseline_style(config)


def test_every_screened_property_can_be_rendered(setup):
    config, _, base = setup
    for prop in load_properties(config):
        assert "render" in prop, f"{prop['id']} is screened in but has no render settings"
    variants(load_properties(config), base)  # every override names a real style parameter


@pytest.mark.parametrize("prop", [p for p in load_properties(load_config(), screened_only=False) if "render" in p],
                         ids=lambda p: p["id"])
def test_low_and_high_clips_differ(setup, prop):
    _, renderer, base = setup
    from render_stimuli import high_levels
    low = render_all(renderer, Style.from_overrides(base, prop["render"]["low"]))
    for suffix, overrides in high_levels(prop).items():
        high = render_all(renderer, Style.from_overrides(base, overrides))
        changed = np.abs(low.astype(int) - high.astype(int)).max(axis=3) > 8
        assert changed.sum() > 50, f"{prop['id']}: low and {suffix} clips look the same"


def test_baseline_leaves_frames_untouched_outside_the_event(setup):
    _, renderer, base = setup
    frames = render_all(renderer, base)
    lead = int(base.lead_time_s * FPS)
    assert np.array_equal(frames[0], background()) or 25 - lead <= 0
    assert not np.array_equal(frames[40], background())


def test_unknown_style_parameter_is_rejected():
    with pytest.raises(ValueError):
        Style.from_overrides(Style(), {"path_colour_of_my_choice": 1})
