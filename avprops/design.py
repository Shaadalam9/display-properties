"""Best-Worst Scaling (MaxDiff) design generation.

Each participant gets one design version: a list of trials, each showing
``items_per_trial`` properties. Within a version every property appears
``appearances_per_item`` times (one more for a few properties when the numbers do
not divide evenly), no property appears twice in a trial, and pairs of properties
are spread as evenly as possible across trials. Versions differ, so pair and
position effects average out across participants.
"""

from __future__ import annotations

import itertools
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _score(blocks: list[list[int]], n_items: int) -> float:
    """Lower is better: imbalance of item counts, then of pair co-occurrence."""
    counts = np.zeros(n_items)
    pairs = np.zeros((n_items, n_items))
    for block in blocks:
        for i in block:
            counts[i] += 1
        for i, j in itertools.combinations(block, 2):
            pairs[i, j] += 1
            pairs[j, i] += 1
    upper = pairs[np.triu_indices(n_items, 1)]
    return 1000.0 * float(counts.max() - counts.min()) + float(upper.var())


def bws_version(n_items: int, items_per_trial: int, appearances_per_item: int,
                rng: np.random.Generator, attempts: int = 60) -> list[list[int]]:
    """Return one balanced BWS design as a list of trials of item indices."""
    if not 2 <= items_per_trial <= n_items:
        raise ValueError(f"items_per_trial must be between 2 and {n_items}")
    if appearances_per_item < 1:
        raise ValueError("appearances_per_item must be at least 1")
    n_trials = math.ceil(n_items * appearances_per_item / items_per_trial)

    best, best_score = None, math.inf
    for _ in range(attempts):
        counts = np.zeros(n_items)
        pairs = np.zeros((n_items, n_items))
        blocks = []
        for _ in range(n_trials):
            block: list[int] = []
            for _ in range(items_per_trial):
                candidates = [i for i in range(n_items) if i not in block]
                # Least-shown items first, then the item that has met the block's
                # members least often; random noise breaks ties.
                cost = [counts[i] * 100 + sum(pairs[i, j] for j in block) + rng.random()
                        for i in candidates]
                block.append(candidates[int(np.argmin(cost))])
            for i in block:
                counts[i] += 1
            for i, j in itertools.combinations(block, 2):
                pairs[i, j] += 1
                pairs[j, i] += 1
            blocks.append(block)
        score = _score(blocks, n_items)
        if score < best_score:
            best, best_score = blocks, score

    assert best is not None
    order = rng.permutation(len(best))
    return [list(rng.permutation(best[t]).tolist()) for t in order]


def ab_scenes(properties: list[dict[str, Any]], scenes: list[dict[str, Any]],
              checked: dict[str, dict[str, dict]] | None) -> dict[str, list[str]]:
    """Scenes in which each property's A/B pair visibly differs, from render_stimuli.py
    --check. Without a check every scene is allowed."""
    scene_ids = [s["id"] for s in scenes]
    if not checked:
        return {p["id"]: scene_ids for p in properties}
    allowed = {}
    for p in properties:
        ok = [s for s in scene_ids if checked.get(s, {}).get(p["id"], {}).get("ok")]
        if not ok:
            raise ValueError(f"{p['id']}: its low and high clips differ too little in every checked scene; "
                             "add a scene that has what it acts on, or screen it out")
        allowed[p["id"]] = ok
    return allowed


def comprehension_plan(scenes: list[dict[str, Any]], allowed: dict[str, list[str]],
                       baseline_scene: str) -> list[dict[str, Any]]:
    """One comprehension question per scene that has one, never the baseline scene (its
    whole clip has been seen by then). Lists the properties whose pair works in it."""
    plan = []
    for scene in scenes:
        spec = scene.get("comprehension")
        if not spec or scene["id"] == baseline_scene:
            continue
        properties = [p for p, ok in allowed.items() if scene["id"] in ok]
        if properties:
            plan.append({"scene": scene["id"], "stop_s": round(spec["stop_s"] - scene["start_s"], 2),
                         "question": spec["question"], "options": spec["options"], "correct": spec["correct"],
                         "properties": properties})
    return plan


def build_design(config: dict[str, Any], properties: list[dict[str, Any]],
                 scenes: list[dict[str, Any]], checked: dict | None = None,
                 questionnaires: list[dict[str, Any]] | None = None,
                 previews: set[str] | None = None) -> dict[str, Any]:
    """Build the design document the participant app loads."""
    if len(properties) < 2:
        raise ValueError("at least two screened-in properties are needed")
    scene_ids = [s["id"] for s in scenes]
    if config["baseline_scene"] not in scene_ids:
        raise ValueError(f"baseline_scene {config['baseline_scene']!r} is not in the scenes file")

    k = min(config["bws_items_per_trial"], len(properties))
    rng = np.random.default_rng(config["bws_design_seed"])
    ids = [p["id"] for p in properties]
    versions = []
    for _ in range(config["bws_design_versions"]):
        blocks = bws_version(len(ids), k, config["bws_appearances_per_item"], rng)
        versions.append([[ids[i] for i in block] for block in blocks])

    pairs = config["ab_pairs_per_participant"]
    allowed = ab_scenes(properties, scenes, checked)
    return {
        "study_id": config["study_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "properties": [
            {key: p[key] for key in ("id", "element", "name", "description", "low", "high")}
            | ({"variants": list(p["render"]["variants"])} if "variants" in p.get("render", {}) else {})
            | ({"preview": f"previews/{p['id']}.mp4"} if p["id"] in (previews or ()) else {})
            for p in properties
        ],
        "scenes": [{"id": s["id"], "name": s["name"], "country": s.get("country", ""),
                    "criticality": s.get("criticality", "")} for s in scenes],
        "baseline_scene": config["baseline_scene"],
        "bws": {"items_per_trial": k, "versions": versions},
        "ab": {
            "pairs_per_participant": len(ids) if pairs in (0, None) else min(pairs, len(ids)),
            "collect_ratings": bool(config["ab_collect_ratings"]),
            "scale_points": int(config["rating_scale_points"]),
            "scenes": allowed,
            "repeats": int(config.get("ab_repeats", 0)),
            "no_display": bool(config.get("ab_no_display", False)),
        },
        "comprehension": (comprehension_plan(scenes, allowed, config["baseline_scene"])
                          if config.get("comprehension") else []),
        "questionnaires": [
            {key: q[key] for key in ("id", "name", "preamble", "labels")}
            | {"items": [{"id": i["id"], "text": i["text"]} for i in q["items"]]}
            for q in (questionnaires or [])
        ],
        "attention_fail_at": int(config["attention_fail_at"]),
        "stimulus_url": config.get("stimulus_url", "stimuli/"),
    }
