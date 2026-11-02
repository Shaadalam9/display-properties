"""Write synthetic participant data with a known ground truth.

    uv run python simulate.py --n 150
    uv run python analysis.py --data data/simulated/entries.jsonl --out _output/simulated

Each property gets a true importance, a probability of being noticed and a population
share preferring its high level. Participants answer the BWS trials of a design
version (best = highest noisy utility, worst = lowest) and the A/B pairs from those
truths. The output has the shape the server stores (``{"data": [jsPsych rows]}`` per
line), so the whole path from raw rows to the selection is checked before any real
data exists. The truth is written to data/simulated/truth.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from avprops.common import load_config, resolve, PROJECT_ROOT

RATING_KEYS = ("understanding", "safety", "load")


TRUST_MODERATED = {"show_when": -1.2, "explanation_detail": -1.0, "highlight_scope": -0.8}
"""Log-odds change in choosing the high level per unit of standardised trust: lower-trust
people prefer more information (Hartwich et al. 2021). Used to check the analysis finds it."""

HIGH_CRITICALITY_EFFECT = {"explanation_detail": 1.5}
"""Log-odds change in choosing the high level in high- vs low-criticality scenes."""

COLOUR_DEPENDENT = ("path_uncertainty", "class_colours")
CVD_NOTICE_FACTOR = 0.3  # people with a colour vision deficiency notice colour-only changes this much as often
COUNTRIES = ("US", "EG", "NL", "IN")


def simulate(design: dict, n: int, seed: int) -> tuple[list[dict], dict]:
    """Return documents ``{"data": rows}`` (one per participant) and the truth."""
    rng = np.random.default_rng(seed)
    props = [p["id"] for p in design["properties"]]
    truth = {
        p: {
            "importance": float(rng.normal(0, 1)),
            "notice_prob": float(rng.uniform(0.5, 0.98)),
            "prefer_high": float(rng.uniform(0.1, 0.9)),
            "trust_effect": TRUST_MODERATED.get(p, 0.0),
            "comprehension_gain": float(rng.uniform(-0.1, 0.3)),
        }
        for p in props
    }
    truth["_global"] = {"prefer_no_display": 0.15, "repeat_consistency": 0.85, "cvd_share": 0.08}
    scene_meta = {sc["id"]: sc for sc in design["scenes"]}
    variants_of = {p["id"]: p["variants"] for p in design["properties"] if p.get("variants")}
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
        version = int(rng.integers(len(design["bws"]["versions"])))
        add("instructions", bws_version=version)
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
        for c in design["comprehension"]:
            p = str(rng.choice(c["properties"]))
            level = "low" if rng.random() < 0.5 else "high"
            correct = rng.random() < 0.6 + (truth[p]["comprehension_gain"] if level == "high" else 0.0)
            add("comprehension", scene=c["scene"], property=p, level=level, expected=c["correct"],
                answer=c["correct"] if correct else "other", correct=bool(correct), rt=4000, hidden=False)
        add("open_question", scene=design["baseline_scene"], response={"open": "Show why it brakes."},
            rt=20000, hidden=False)
        for t, items in enumerate(design["bws"]["versions"][version]):
            noisy = {p: utility[p] + rng.gumbel() for p in items}
            add("bws", trial=t, items=items, best=max(noisy, key=noisy.get), worst=min(noisy, key=noisy.get),
                rt=int(rng.integers(2500, 12000)), hidden=False)

        def pair(index: int, prop: str, scene: str, level: str, noticed: bool, repeat: bool = False,
                 levels: tuple[str, str] = ("low", "high")) -> None:
            a_level = levels[0] if rng.random() < 0.5 else levels[1]
            b_level = levels[1] if a_level == levels[0] else levels[0]
            variants = variants_of.get(prop)
            add("ab", index=index, property=prop, scene=scene, a_level=a_level, b_level=b_level, noticed=noticed,
                high_variant=str(rng.choice(variants)) if variants else None,
                preferred_option="A" if level == a_level else "B", preferred_level=level, repeat=repeat,
                importance=int(np.clip(round((points + 1) / 2 + 1.2 * utility.get(prop, 0)), 1, points)),
                ratings={lvl: {key: int(rng.integers(1, points + 1)) for key in RATING_KEYS} for lvl in levels},
                rt=int(rng.integers(8000, 40000)), hidden=False)

        answered = []
        for k, p in enumerate(rng.permutation(props)[: design["ab"]["pairs_per_participant"]]):
            p = str(p)
            scene = str(rng.choice(design["ab"]["scenes"][p]))
            notice_prob = truth[p]["notice_prob"] * (CVD_NOTICE_FACTOR if cvd and p in COLOUR_DEPENDENT else 1.0)
            noticed = bool(rng.random() < notice_prob)
            base = np.log(truth[p]["prefer_high"] / (1 - truth[p]["prefer_high"]))
            critical = scene_meta.get(scene, {}).get("criticality") == "high"
            logit = base + truth[p]["trust_effect"] * trust + (HIGH_CRITICALITY_EFFECT.get(p, 0.0) if critical else 0.0)
            prob_high = 1 / (1 + np.exp(-logit))
            level = ("high" if rng.random() < prob_high else "low") if noticed \
                else ("high" if rng.random() < 0.5 else "low")
            pair(k, p, scene, level, noticed)
            answered.append((p, scene, level, noticed))
        if design["ab"]["no_display"]:
            none = rng.random() < truth["_global"]["prefer_no_display"]
            pair(len(answered), "no_display", design["scenes"][0]["id"], "none" if none else "baseline", True,
                 levels=("none", "baseline"))
        for j, (p, scene, level, noticed) in enumerate(answered[: design["ab"]["repeats"]]):
            same = rng.random() < truth["_global"]["repeat_consistency"]
            pair(len(answered) + 1 + j, p, scene, level if same else ("low" if level == "high" else "high"),
                 noticed, repeat=True)
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
    documents, truth = simulate(design, args.n, args.seed)

    out = PROJECT_ROOT / "data" / "simulated"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "entries.jsonl").open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc) + "\n")
    (out / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(f"Wrote {len(documents)} participants to {out / 'entries.jsonl'} and the truth to {out / 'truth.json'}")


if __name__ == "__main__":
    main()
