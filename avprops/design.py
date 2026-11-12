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


def limit_scenes(allowed: dict[str, list[str]], scenes: list[dict[str, Any]],
                 checked: dict[str, dict[str, dict]] | None, k: int) -> dict[str, list[str]]:
    """At most k scenes per property (0 = no limit), so fewer clips have to be hosted. The
    scenes are picked one by one: first the one where the pair differs most, then each time
    the scene that best balances calm and risky and adds a new country, ties broken by how
    much the pair differs there."""
    if not k:
        return allowed
    meta = {s["id"]: s for s in scenes}
    changed = lambda p, s: (checked or {}).get(s, {}).get(p, {}).get("changed", 0.0)  # noqa: E731
    limited = {}
    for prop, ok in allowed.items():
        chosen: list[str] = []
        pool = sorted(ok, key=lambda s: -changed(prop, s))
        while pool and len(chosen) < k:
            risky = sum(meta[s].get("criticality") == "high" for s in chosen)
            want_risky = risky <= len(chosen) - risky
            countries = {meta[s].get("country") for s in chosen}
            best = max(pool, key=lambda s: ((meta[s].get("criticality") == "high") == want_risky,
                                           meta[s].get("country") not in countries, changed(prop, s)))
            chosen.append(best)
            pool.remove(best)
        limited[prop] = [s for s in ok if s in chosen]  # keep the scenes file's order
    return limited


def comprehension_plan(scenes: list[dict[str, Any]], allowed: dict[str, list[str]],
                       baseline_scene: str, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    """One comprehension question per scene that has one, never the baseline scene (its
    whole clip has been seen by then). Lists the properties whose pair works in it, except
    ``excluded``: those whose high level names the car's action in words and so would give
    the answer away before the clip stops."""
    plan = []
    for scene in scenes:
        spec = scene.get("comprehension")
        if not spec or scene["id"] == baseline_scene:
            continue
        properties = [p for p, ok in allowed.items() if scene["id"] in ok and p not in (excluded or set())]
        if properties:
            plan.append({"scene": scene["id"], "stop_s": round(spec["stop_s"] - scene["start_s"], 2),
                         "question": spec["question"], "options": spec["options"], "correct": spec["correct"],
                         "properties": properties})
    return plan


def ab_counts(config: dict[str, Any], properties: list[dict[str, Any]]) -> tuple[int, int]:
    """Tunable and categorical properties each participant compares (capped at how many exist)."""
    from avprops.common import TUNABLE
    tunable = sum(p.get("type") in TUNABLE for p in properties)
    return (min(int(config["ab_tunable_per_participant"]), tunable),
            min(int(config["ab_categorical_per_participant"]), len(properties) - tunable))


LEVELS = ("low", "high", "none", "baseline")  # level codes in the participant plans


def build_plans(config: dict[str, Any], design: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    """The participant plans: every random assignment of a session, made once here, so the
    experiment and simulate.py use the same ones (review by Mark Colley). Each participant
    gets one plan (picked from their code); the plan is logged with their data.

    A plan is ``{"bws_version": v, "pairs": [...], "comprehension": [...]}``, compact because
    every participant downloads all of them:

    * pair = [property index, scene index, A level, B level, variant index, catch], with the
      indices into design["properties"] and design["scenes"], levels as indices into LEVELS,
      variant -1 when the property has none, catch 0/1; property index -1 is the display vs
      no display pair (levels "none"/"baseline");
    * comprehension = [property index, level, variant index], one per design["comprehension"]
      entry, in that order.

    The pairs: ab.tunable_per_participant tunable properties, each in ab.scenes_per_property
    different scenes, and ab.categorical_per_participant categorical ones in one scene; the low
    and high level on random sides; one palette variant per property for all its scenes;
    ab.catch_pairs pairs of one clip on both sides; the display vs no display pair; in random
    order with no property's two pairs one after the other."""
    props = design["properties"]
    index = {p["id"]: k for k, p in enumerate(props)}
    scene_index = {s["id"]: k for k, s in enumerate(design["scenes"])}
    ab = design["ab"]
    tunable = [p["id"] for p in props if p.get("type") != "categorical"]
    categorical = [p["id"] for p in props if p.get("type") == "categorical"]

    def variant(prop: str) -> int:
        variants = props[index[prop]].get("variants")
        return int(rng.integers(len(variants))) if variants else -1

    plans = []
    for _ in range(int(config.get("participant_plans", 500))):
        pairs = []
        compared = [(str(p), ab["scenes_per_property"]) for p in rng.permutation(tunable)[:ab["tunable_per_participant"]]] + \
                   [(str(p), 1) for p in rng.permutation(categorical)[:ab["categorical_per_participant"]]]
        for prop, n_scenes in compared:
            v = variant(prop)
            for scene in rng.permutation(ab["scenes"][prop])[:n_scenes]:
                a_high = int(rng.random() < 0.5)
                pairs.append([index[prop], scene_index[str(scene)], a_high, 1 - a_high, v, 0])
        for _ in range(ab["catch_pairs"]):
            prop = str(rng.choice([p["id"] for p in props]))
            level = int(rng.random() < 0.5)
            pairs.append([index[prop], scene_index[str(rng.choice(ab["scenes"][prop]))], level, level, variant(prop), 1])
        if ab["no_display"]:
            a_none = rng.random() < 0.5
            pairs.append([-1, int(rng.integers(len(design["scenes"]))), 2 if a_none else 3, 3 if a_none else 2, -1, 0])
        for _ in range(200):
            order = [pairs[k] for k in rng.permutation(len(pairs))]
            if not any(a[0] == b[0] >= 0 and not a[5] and not b[5] for a, b in zip(order, order[1:])):
                break
        comprehension = []
        for c in design["comprehension"]:
            prop = str(rng.choice(c["properties"]))
            comprehension.append([index[prop], int(rng.random() < 0.5), variant(prop)])
        plans.append({"bws_version": int(rng.integers(len(design["bws"]["versions"]))), "pairs": order,
                      "comprehension": comprehension})
    return plans


def expand_plan(plan: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    """A plan in words: the same structure the experiment logs in its first instructions row."""
    props, scenes = design["properties"], design["scenes"]

    def variant_name(k: int, v: int) -> str | None:
        return props[k]["variants"][v] if k >= 0 and v >= 0 else None

    pairs = [{"property": "no_display" if k < 0 else props[k]["id"], "scene": scenes[s]["id"],
              "a_level": LEVELS[a], "b_level": LEVELS[b], "high_variant": variant_name(k, v), "catch": bool(c)}
             for k, s, a, b, v, c in plan["pairs"]]
    comprehension = [{"scene": spec["scene"], "property": props[k]["id"], "level": LEVELS[lv],
                      "high_variant": variant_name(k, v)}
                     for spec, (k, lv, v) in zip(design["comprehension"], plan["comprehension"])]
    return {"bws_version": plan["bws_version"], "pairs": pairs, "comprehension": comprehension}


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

    allowed = limit_scenes(ab_scenes(properties, scenes, checked), scenes, checked,
                           config.get("ab_max_scenes_per_property", 0))
    few = [p for p, s in allowed.items() if len(s) < int(config["ab_scenes_per_property"])]
    if few:
        raise ValueError(f"each property is compared in {config['ab_scenes_per_property']} scenes, but these pass the "
                         f"check in fewer: {', '.join(few)}")
    design = {
        "study_id": config["study_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "properties": [
            {key: p[key] for key in ("id", "type", "element", "name", "description", "low", "high")}
            | ({"variants": list(p["render"]["variants"])} if "variants" in p.get("render", {}) else {})
            | ({"preview": f"previews/{p['id']}.mp4"} if p["id"] in (previews or ()) else {})
            for p in properties
        ],
        "scenes": [{"id": s["id"], "name": s["name"], "country": s.get("country", ""),
                    "criticality": s.get("criticality", "")} for s in scenes],
        "baseline_scene": config["baseline_scene"],
        "bws": {"items_per_trial": k, "versions": versions},
        "ab": {
            # each participant compares some tunable properties, each in scenes_per_property
            # different scenes (the agreement between a participant's answers gives the
            # heterogeneity H), some categorical ones in one scene (they are settled by the
            # majority, so need no H), and catch pairs of two identical clips (false alarms on
            # "did you notice a difference")
            "tunable_per_participant": ab_counts(config, properties)[0],
            "categorical_per_participant": ab_counts(config, properties)[1],
            "scenes_per_property": int(config["ab_scenes_per_property"]),
            "catch_pairs": int(config.get("ab_catch_pairs", 0)),
            "collect_ratings": bool(config["ab_collect_ratings"]),
            "scale_points": int(config["rating_scale_points"]),
            "scenes": allowed,
            "no_display": bool(config.get("ab_no_display", False)),
            # seconds before Continue works: the clips have to be watched once
            "min_view_s": float(config.get("ab_min_view_s", 0)),
        },
        "comprehension": (comprehension_plan(scenes, allowed, config["baseline_scene"],
                                             {p["id"] for p in properties if p.get("in_comprehension") is False})
                          if config.get("comprehension") else []),
        "questionnaires": [
            {key: q[key] for key in ("id", "name", "preamble", "labels")}
            | {"items": [{"id": i["id"], "text": i["text"]} for i in q["items"]]}
            for q in (questionnaires or [])
        ],
        "attention_fail_at": int(config["attention_fail_at"]),
        "stimulus_url": config.get("stimulus_url", "stimuli/"),
    }
    design["plans"] = build_plans(config, design, np.random.default_rng(config["bws_design_seed"] + 1))
    return design
