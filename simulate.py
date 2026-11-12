"""Write synthetic participant data with a known ground truth.

    uv run python simulate.py --n 150
    uv run python analysis.py --data data/simulated/entries.jsonl --out _output/simulated

Each property gets a true importance (drives the BWS answers and the importance rating), a
probability of being noticed, and a preference structure of one of three kinds:

* consensus: nearly everyone leans the same way;
* types: two stable groups of people who prefer opposite levels (what personalisation is for);
* indifferent: nobody cares, so answers are noise around "no preference".

A participant's answer in a scene is their own lean on the property plus noise from scene to
scene, cut into the graded scale (-2 .. 2 towards the high level). Some participants are
indifferent responders who answer around "no preference" whatever they see, two properties
share each participant's lean (a redundant pair), and catch pairs of two identical clips draw
false alarms. The truth (noticing after false alarms, expected BWS score, heterogeneity H and
the properties the rule should select) is computed from a large virtual population, so a
simulated study's selection can be compared with it (``true_selection``). The output has the
shape the server stores (``{"data": [jsPsych rows]}`` per line); the truth is written to
data/simulated/truth.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from avprops.common import load_config, resolve, PROJECT_ROOT
from avprops.design import expand_plan

RATING_KEYS = ("understanding", "safety", "load")


TRUST_MODERATED = {"show_when": -1.2, "explanation_detail": -1.0, "highlight_scope": -0.8}
"""Log-odds change in choosing the high level per unit of standardised trust: lower-trust
people prefer more information (Hartwich et al. 2021). Used to check the analysis finds it."""

HIGH_CRITICALITY_EFFECT = {"explanation_detail": 1.5}
"""Log-odds change in choosing the high level in high- vs low-criticality scenes."""

COLOUR_DEPENDENT = ("path_uncertainty", "class_colours", "palette_id")
CVD_NOTICE_FACTOR = 0.3  # people with a colour vision deficiency notice colour-only changes this much as often
COUNTRIES = ("US", "EG", "NL", "IN")
SCENE_NOISE = 0.6          # sd of a participant's answer from scene to scene, on the latent scale
INDIFFERENT_SHARE = 0.15   # participants who answer around "no preference" whatever they see
FALSE_ALARM = 0.10         # chance of reporting a difference that is not there
KINDS = ("consensus", "types", "indifferent")


def graded(latent: float) -> int:
    """Latent lean towards the high level -> -2 .. 2."""
    return int(np.clip(np.round(latent / 0.8), -2, 2))


def person_lean(rng: np.random.Generator, truth_p: dict) -> float:
    """A participant's own, stable lean towards the high level on one property."""
    kind = truth_p["kind"]
    if kind == "consensus":
        return truth_p["mean"] + rng.normal(0, 0.3)
    if kind == "types":
        return truth_p["mean"] if rng.random() < truth_p["share_high"] else -truth_p["mean"]
    return 0.0


def answer(rng: np.random.Generator, lean: float, noticed: bool, indifferent: bool) -> int:
    """Graded answer in one scene."""
    if indifferent or not noticed:
        return 0 if rng.random() < 0.6 else int(rng.choice([-1, 1]))
    return graded(lean + rng.normal(0, SCENE_NOISE))


def make_truth(props: list[str], rng: np.random.Generator) -> dict:
    truth = {}
    for p in props:
        kind = str(rng.choice(KINDS, p=[0.35, 0.35, 0.30]))
        sign = 1 if rng.random() < 0.5 else -1
        truth[p] = {
            "importance": float(rng.normal(0, 1)),
            "notice_prob": float(rng.uniform(0.5, 0.98)),
            "kind": kind,
            "mean": float(sign * rng.uniform(0.8, 1.6)) if kind == "consensus" else float(rng.uniform(0.9, 1.6)),
            "share_high": float(rng.uniform(0.3, 0.7)),
            "trust_effect": TRUST_MODERATED.get(p, 0.0),
            "comprehension_gain": float(rng.uniform(-0.1, 0.3)),
        }
    return truth


def population_truth(design: dict, truth: dict, config: dict, rng: np.random.Generator, n: int = 4000) -> None:
    """Add to the truth what a study of infinite size would find: noticing after false alarms,
    the expected BWS score and H; and the properties the rule should select."""
    props = [p["id"] for p in design["properties"]]
    kind_of = {p["id"]: p.get("type") for p in design["properties"]}
    shared = truth["_global"]["redundant"]
    bws_sum, bws_n = {p: 0.0 for p in props}, {p: 0 for p in props}
    versions = design["bws"]["versions"]
    for i in range(n):
        utility = {p: truth[p]["importance"] + rng.normal(0, 0.7) for p in props}
        shown = {p: [0, 0, 0] for p in props}
        for items in versions[i % len(versions)]:
            noisy = {p: utility[p] + rng.gumbel() for p in items}
            for p in items:
                shown[p][2] += 1
            shown[max(noisy, key=noisy.get)][0] += 1
            shown[min(noisy, key=noisy.get)][1] += 1
        for p, (best, worst, k) in shown.items():
            if k:
                bws_sum[p] += (best - worst) / k
                bws_n[p] += 1
    scene_meta = {s["id"]: s for s in design["scenes"]}
    for p in props:
        # the same answer model as in simulate(): own lean, trust effect, and the effect of risky
        # scenes, in two of the property's scenes
        scenes = design["ab"]["scenes"][p]
        first, second = [], []
        for _ in range(n):
            lean = person_lean(rng, truth[p]) + truth[p]["trust_effect"] * rng.normal(0, 1)
            indifferent = rng.random() < INDIFFERENT_SHARE
            answers = []
            for scene in rng.permutation(scenes)[:2]:
                critical = scene_meta.get(str(scene), {}).get("criticality") == "high"
                shift = HIGH_CRITICALITY_EFFECT.get(p, 0.0) if critical else 0.0
                answers.append(answer(rng, lean + shift, rng.random() < truth[p]["notice_prob"], indifferent))
            first.append(answers[0])
            second.append(answers[1])
        from avprops.analysis import icc_two
        both = [(x, y) for x, y in zip(first, second) if x != 0 and y != 0]  # as analysis._two_answers
        truth[p]["H"] = icc_two(*(np.sign(np.array(v, float)) for v in zip(*both)))
        truth[p]["bws_score"] = bws_sum[p] / max(bws_n[p], 1)
        # reported noticing = noticed + false alarms among the rest; minus the false-alarm rate:
        truth[p]["noticed_corrected"] = truth[p]["notice_prob"] * (1 - FALSE_ALARM)
    from avprops.analysis import _selection
    import pandas as pd
    tunable = {p for p in props if kind_of[p] in ("continuous", "ordinal")}
    chosen, _ = _selection(pd.Series({p: truth[p]["bws_score"] for p in props}),
                           pd.Series({p: truth[p]["noticed_corrected"] for p in props}),
                           pd.Series({p: truth[p]["H"] for p in props}), tunable, config)
    truth["_global"]["selected"] = chosen
    truth["_global"]["redundant"] = shared


def true_selection(truth: dict) -> list[str]:
    return truth["_global"]["selected"]


def simulate(design: dict, n: int, seed: int, config: dict | None = None,
             truth: dict | None = None) -> tuple[list[dict], dict]:
    """Return documents ``{"data": rows}`` (one per participant) and the truth. Pass a
    ``truth`` from an earlier call to simulate more studies of the same population."""
    rng = np.random.default_rng(seed)
    config = config or load_config()
    props = [p["id"] for p in design["properties"]]
    if truth is None:
        truth = make_truth(props, rng)
        tunable = [p["id"] for p in design["properties"] if p.get("type") in ("continuous", "ordinal")]
        redundant = [str(x) for x in rng.choice(tunable, 2, replace=False)]
        truth[redundant[1]].update({k: truth[redundant[0]][k] for k in ("kind", "mean", "share_high")})
        truth["_global"] = {"prefer_no_display": 0.15, "cvd_share": 0.08, "false_alarm": FALSE_ALARM,
                            "indifferent_share": INDIFFERENT_SHARE, "redundant": redundant}
        population_truth(design, truth, config, np.random.default_rng(seed + 10_000))
    shared = truth["_global"]["redundant"]
    scene_meta = {sc["id"]: sc for sc in design["scenes"]}
    points = design["ab"]["scale_points"]
    documents = []
    for i in range(n):
        code = f"sim{i:04d}"
        rows: list[dict] = []

        def add(task: str, **fields) -> None:
            rows.append({"trial_index": len(rows), "worker_code": code, "study_id": design["study_id"],
                         "design_generated_at": design["generated_at"], "debug": False, "task": task, **fields})

        trust = float(rng.normal(0, 1))  # standardised propensity to trust
        utility = {p: truth[p]["importance"] + rng.normal(0, 0.7) for p in props}
        # the session's assignments come from the same plans the experiment uses
        plan_index = int(rng.integers(len(design["plans"])))
        plan = expand_plan(design["plans"][plan_index], design)
        version = plan["bws_version"]
        add("instructions", plan_index=plan_index, bws_version=version, ab_plan=plan["pairs"],
            comprehension_plan=plan["comprehension"])
        add("consent", response={"consent": "Yes"}, rt=3000)
        country = str(rng.choice(COUNTRIES))
        cvd = bool(rng.random() < truth["_global"]["cvd_share"])
        add("demographics", response={"age": "30"})
        add("demographics", response={"nationality": country, "country_residence": country})
        add("demographics", response={"gender": "Female", "colour_vision_deficiency": "Yes" if cvd else "No"})
        for q in design["questionnaires"]:
            n_labels = len(q["labels"])
            answers = {}
            for item in q["items"]:
                # trust drives the trust questionnaire; reversed items are answered the other way
                lean = trust if q["id"] == "propensity_to_trust" else rng.normal(0, 1)
                lean = -lean if item["id"] in ("ptt_1", "ati_3", "ati_6", "ati_8") else lean
                value = (n_labels - 1) / 2 + lean * (n_labels - 1) / 4 + rng.normal(0, 0.5)
                answers[item["id"]] = int(np.clip(round(value), 0, n_labels - 1))
            add("questionnaire", questionnaire=q["id"], response=answers)
        for c, drawn in zip(design["comprehension"], plan["comprehension"]):
            p, level = drawn["property"], drawn["level"]
            correct = rng.random() < 0.6 + (truth[p]["comprehension_gain"] if level == "high" else 0.0)
            add("comprehension", scene=c["scene"], property=p, level=level, high_variant=drawn["high_variant"],
                expected=c["correct"], answer=c["correct"] if correct else "other", correct=bool(correct),
                rt=4000, hidden=False)
        add("open_question", scene=design["baseline_scene"], response={"open": "Show why it brakes."},
            rt=20000, hidden=False)
        for t, items in enumerate(design["bws"]["versions"][version]):
            noisy = {p: utility[p] + rng.gumbel() for p in items}
            add("bws", trial=t, items=items, best=max(noisy, key=noisy.get), worst=min(noisy, key=noisy.get),
                rt=int(rng.integers(2500, 12000)), hidden=False)

        indifferent = bool(rng.random() < truth["_global"]["indifferent_share"])
        lean = {p: person_lean(rng, truth[p]) + truth[p]["trust_effect"] * trust for p in props}
        lean[shared[1]] = lean[shared[0]]  # the redundant pair: one lean for both

        def pair(index: int, prop: str, scene: str, a_level: str, b_level: str, towards_b: int, noticed: bool,
                 catch: bool = False, variant: str | None = None) -> None:
            if prop == "no_display":
                b_up = b_level == "baseline"
            else:
                b_up = b_level == "high"
            level = a_level if towards_b < 0 else (b_level if towards_b > 0 else "none")
            add("ab", index=index, property=prop, scene=scene, a_level=a_level, b_level=b_level, noticed=noticed,
                high_variant=variant, catch=catch,
                preferred_option="A" if towards_b < 0 else ("B" if towards_b > 0 else "none"),
                preferred_level=level, preference=None if catch else (towards_b if b_up else -towards_b),
                importance=int(np.clip(round((points + 1) / 2 + 1.2 * utility.get(prop, 0)), 1, points)),
                rt=int(rng.integers(8000, 40000)), hidden=False)

        for k, pair_plan in enumerate(plan["pairs"]):
            p, scene, a_level, b_level = pair_plan["property"], pair_plan["scene"], pair_plan["a_level"], pair_plan["b_level"]
            if p == "no_display":
                wants = -1 if rng.random() < truth["_global"]["prefer_no_display"] else 1
                pair(k, p, scene, a_level, b_level, wants if b_level == "baseline" else -wants, True)
            elif pair_plan["catch"]:
                noticed = bool(rng.random() < truth["_global"]["false_alarm"])
                pair(k, p, scene, a_level, b_level, 0 if rng.random() < 0.7 else int(rng.choice([-1, 1])), noticed,
                     catch=True, variant=pair_plan["high_variant"])
            else:
                notice_prob = truth[p]["notice_prob"] * (CVD_NOTICE_FACTOR if cvd and p in COLOUR_DEPENDENT else 1.0)
                saw = bool(rng.random() < notice_prob)
                noticed = saw or bool(rng.random() < truth["_global"]["false_alarm"])
                critical = scene_meta.get(scene, {}).get("criticality") == "high"
                towards_high = answer(rng, lean[p] + (HIGH_CRITICALITY_EFFECT.get(p, 0.0) if critical else 0.0),
                                      saw, indifferent)
                pair(k, p, scene, a_level, b_level, towards_high if b_level == "high" else -towards_high, noticed,
                     variant=pair_plan["high_variant"])
        for _ in range(5):
            add("attention", correct=bool(rng.random() < 0.97))
        add("final", response={"comment": ""})
        documents.append({"data": rows})
    return documents, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=150, help="number of synthetic participants")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    config = load_config()
    design = json.loads(resolve(config["design_output"]).read_text(encoding="utf-8"))
    documents, truth = simulate(design, args.n, args.seed, config)

    out = PROJECT_ROOT / "data" / "simulated"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "entries.jsonl").open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc) + "\n")
    (out / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(f"Wrote {len(documents)} participants to {out / 'entries.jsonl'} and the truth to {out / 'truth.json'}")


if __name__ == "__main__":
    main()
