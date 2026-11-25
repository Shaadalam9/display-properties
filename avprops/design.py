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
    experiment and simulate.py use the same ones (review by Mark Colley). The server hands the
    plans out in turn (plan 0, 1, 2, ...), so every consecutive run of participants is balanced;
    the plan is logged with their data.

    A plan is ``{"bws_version": v, "pairs": [...], "comprehension": [...]}``, compact because
    every participant downloads all of them:

    * pair = [property index, scene index, A level, B level, variant index, catch], with the
      indices into design["properties"] and design["scenes"], levels as indices into LEVELS,
      variant -1 when the property has none, catch 0/1; property index -1 is the display vs
      no display pair (levels "none"/"baseline");
    * comprehension = [property index, level, variant index], one per design["comprehension"]
      entry, in that order (the experiment shows them in random order).

    Balanced across plans, as a counterbalanced design is (not left to independent draws):

    * the BWS versions in blocks: every run of as many plans as there are versions uses each
      version once, in random order;
    * which properties a participant compares, the scenes each is compared in, the palette
      variant, the catch pairs' property, scene and level, the no-display pair's scene and the
      comprehension clips' property and level: each time the least used so far, ties at random;
    * sides: within a participant, a tunable property is high on A in one scene and high on B in
      the other, and about half of the single pairs (categorical, display vs no display) have
      it on A, given to the properties that had it least often; so half of each participant's
      pairs have the high level (or the display) on A, and each property is on A about half
      the time across participants.

    Order within a plan: random, with no property's two pairs one after the other, the catch
    pairs not among the first two and not next to each other, and no scene twice in a row
    (dropped, for that plan only, when its scenes make it impossible)."""
    props = design["properties"]
    index = {p["id"]: k for k, p in enumerate(props)}
    scene_index = {s["id"]: k for k, s in enumerate(design["scenes"])}
    ab = design["ab"]
    tunable = [p["id"] for p in props if p.get("type") != "categorical"]
    categorical = [p["id"] for p in props if p.get("type") == "categorical"]
    n_versions = len(design["bws"]["versions"])
    used: dict[Any, int] = {}

    def least_used(options: list, key, k: int = 1) -> list:
        """k options (distinct) that were used least so far under ``key``, ties at random."""
        ranked = sorted(rng.permutation(len(options)).tolist(), key=lambda i: used.get((key, _hashable(options[i])), 0))
        picked = [options[i] for i in ranked[:k]]
        for o in picked:
            used[(key, _hashable(o))] = used.get((key, _hashable(o)), 0) + 1
        return picked

    def variant(prop: str, key: str = "variant") -> int:
        variants = props[index[prop]].get("variants")
        return least_used(list(range(len(variants))), (key, prop))[0] if variants else -1

    def good_order(order: list, level: int) -> bool:
        """level 2: all rules; 1: a scene may repeat; 0: only a property's pairs kept apart."""
        real = lambda p: p[0] >= 0 and not p[5]  # noqa: E731
        if any(real(a) and real(b) and a[0] == b[0] for a, b in zip(order, order[1:])):
            return False
        catch = [k for k, p in enumerate(order) if p[5]]
        if level >= 1 and not (all(k >= 2 for k in catch) and not any(b - a == 1 for a, b in zip(catch, catch[1:]))):
            return False
        return level < 2 or not any(a[1] == b[1] for a, b in zip(order, order[1:]))

    version_queue: list[int] = []
    plans = []
    for _ in range(int(config.get("participant_plans", 500))):
        if not version_queue:
            version_queue = rng.permutation(n_versions).tolist()
        pairs = []
        compared = [(p, ab["scenes_per_property"]) for p in least_used(tunable, "tunable", ab["tunable_per_participant"])] + \
                   [(p, 1) for p in least_used(categorical, "categorical", ab["categorical_per_participant"])]
        for prop, n_scenes in compared:
            v = variant(prop)
            scenes = least_used(ab["scenes"][prop], ("scene", prop), n_scenes)
            if n_scenes > 1:
                sides = [k % 2 for k in range(n_scenes)]  # high on A in half of the property's scenes
                sides = [sides[i] for i in rng.permutation(n_scenes)]
            else:
                sides = [None]  # set below, with the participant's other single pairs
            for scene, a_high in zip(scenes, sides):
                pairs.append([index[prop], scene_index[scene], a_high, None if a_high is None else 1 - a_high, v, 0])
        for _ in range(ab["catch_pairs"]):
            prop = least_used([p["id"] for p in props], "catch")[0]
            scene = least_used(ab["scenes"][prop], ("catch_scene", prop))[0]
            level = least_used([0, 1], ("catch_level", prop))[0]
            pairs.append([index[prop], scene_index[scene], level, level, variant(prop, "catch_variant"), 1])
        if ab["no_display"]:
            scene = least_used([s["id"] for s in design["scenes"]], "no_display")[0]
            pairs.append([-1, scene_index[scene], None, None, -1, 0])
        # the single pairs (categorical properties, display vs no display): about half with the
        # high level (or the display) on A, so the participant's pairs are balanced too; which of
        # them get A goes to the properties that had it least often across participants
        single = [p for p in pairs if p[2] is None]
        if single:
            n_a = least_used([len(single) // 2, (len(single) + 1) // 2], ("single_on_a", len(single)))[0]
            share = lambda i: used.get(("side", single[i][0]), 0) / max(used.get(("side_n", single[i][0]), 0), 1)  # noqa: E731
            ranked = sorted(rng.permutation(len(single)).tolist(), key=share)
            for rank, i in enumerate(ranked):
                on_a = rank < n_a
                used[("side", single[i][0])] = used.get(("side", single[i][0]), 0) + on_a
                used[("side_n", single[i][0])] = used.get(("side_n", single[i][0]), 0) + 1
                if single[i][0] < 0:
                    single[i][2], single[i][3] = (3, 2) if on_a else (2, 3)
                else:
                    single[i][2], single[i][3] = int(on_a), 1 - int(on_a)
        order = None
        for level in (2, 1, 0):
            for _ in range(2000):
                candidate = [pairs[k] for k in rng.permutation(len(pairs))]
                if good_order(candidate, level):
                    order = candidate
                    break
            if order:
                break
        if order is None:
            raise ValueError("no order of the A/B pairs keeps a property's two pairs apart")
        comprehension = []
        for c in design["comprehension"]:
            prop = least_used(c["properties"], ("comprehension", c["scene"]))[0]
            level = least_used([0, 1], ("comprehension_level", c["scene"]))[0]
            comprehension.append([index[prop], level, variant(prop, "comprehension_variant")])
        plans.append({"bws_version": int(version_queue.pop(0)), "pairs": order, "comprehension": comprehension})
    return plans


def _hashable(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


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
