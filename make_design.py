"""Generate public/design.json from the screened properties, the scenes and the config.

Run after every change to study/properties.json, study/scenes.json or the config:

    uv run python make_design.py

It also lists the stimulus clips the design needs and which of them are missing.
"""

from __future__ import annotations

import json

from avprops.common import load_config, load_properties, load_questionnaires, load_scenes, resolve, PROJECT_ROOT
from avprops.design import build_design


def required_clips(design: dict) -> list[str]:
    clips = [f"stimuli/{design['baseline_scene']}/baseline.mp4"]
    if design["ab"]["no_display"]:
        for scene in design["scenes"]:
            clips += [f"stimuli/{scene['id']}/baseline.mp4", f"stimuli/{scene['id']}/none.mp4"]
    for prop in design["properties"]:
        highs = [f"high_{v}" for v in prop["variants"]] if prop.get("variants") else ["high"]
        for scene in design["ab"]["scenes"][prop["id"]]:
            for level in ["low", *highs]:
                clips.append(f"stimuli/{scene}/{prop['id']}_{level}.mp4")
    return clips


def main() -> None:
    config = load_config()
    properties = load_properties(config)
    check_path = PROJECT_ROOT / "study" / "stimulus_check.json"
    checked = json.loads(check_path.read_text()) if check_path.exists() else None
    scenes = [s for s in load_scenes(config) if s.get("clip")] if checked else load_scenes(config)
    preview_dir = resolve(config["design_output"]).parent / "stimuli" / "previews"
    previews = {f.stem for f in preview_dir.glob("*.mp4")} if preview_dir.exists() else set()
    design = build_design(config, properties, scenes, checked, load_questionnaires(config), previews)

    out = resolve(config["design_output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(design, indent=1), encoding="utf-8")

    n_trials = len(design["bws"]["versions"][0])
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")
    print(f"  {len(properties)} properties, {len(design['scenes'])} scenes")
    print(f"  BWS: {len(design['bws']['versions'])} versions x {n_trials} trials "
          f"of {design['bws']['items_per_trial']} properties")
    ab = design["ab"]
    print(f"  A/B pairs per participant: {ab['tunable_per_participant']} tunable properties x "
          f"{ab['scenes_per_property']} scenes + {ab['categorical_per_participant']} categorical + "
          f"{ab['catch_pairs']} catch pairs" + (" + 1 display vs no display" if ab["no_display"] else ""))
    print(f"  Comprehension questions: {len(design['comprehension'])} "
          f"({', '.join(c['scene'] for c in design['comprehension'])})")
    print(f"  Questionnaires: {', '.join(q['id'] for q in design['questionnaires']) or 'none'}")
    print(f"  Ranking previews: {sum('preview' in p for p in design['properties'])} of {len(design['properties'])}")
    print("  " + ("Scenes per property from study/stimulus_check.json" if checked
                  else "No stimulus check yet: every scene allowed for every property"))
    for prop, allowed in design["ab"]["scenes"].items():
        if checked and len(allowed) < len(scenes):
            print(f"    {prop}: {', '.join(allowed)}")

    clips = sorted(set(required_clips(design)))
    missing = [c for c in clips if not (out.parent / c).exists()]
    print(f"  Stimulus clips: {len(clips) - len(missing)} of {len(clips)} present in public/stimuli")
    if missing:
        print("  The app shows a placeholder for each missing clip. First missing:")
        for clip in missing[:5]:
            print(f"    public/{clip}")


if __name__ == "__main__":
    main()
