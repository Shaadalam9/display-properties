"""Analysis of the participant records and the pre-registered selection rules.

Inputs are participant records built from the crowdsourced rows by avprops.crowd.
The steps mirror the study design:

1. Exclusions: debug sessions, other studies, unfinished sessions, finished records that
   lack trials, later sessions from an address whose earlier session ended, too many wrong
   attention answers, implausibly fast BWS answers, A/B pairs answered with the tab hidden
   too often. The analysis stops when the kept participants saw different versions of the
   design, unless ``design_versions`` names the version(s) to analyse.
2. Task A (BWS): per participant, (times best - times worst) / times shown for each
   property; the property's score is the mean over participants with a bootstrap CI.
3. Task B (A/B pairs): share who noticed the difference; among those who noticed,
   the share preferring the high level and the majority share; mean importance; and
   the high-minus-low difference of each rating with a Wilcoxon signed-rank test.
4. Selection: noticeable -> matters -> people disagree -> not redundant, then the top
   ``select_n`` by BWS score. Every property gets a status saying where it stopped.

Alongside the selection:
* consistency: share of repeated A/B pairs (sides swapped) answered with the same
  preferred level as the first time; low consistency means apparent disagreement may
  be noise. ``min_repeat_consistency`` can exclude participants below it;
* display vs no display: share of participants who preferred no display at all;
* comprehension: per property, the share answering "what will the car do next?"
  correctly after its low vs its high clip (between participants, Fisher's exact test);
* questionnaires: per participant scores, and per property whether choosing the high
  level goes with the score (point-biserial r), i.e. whether preferences split by user type;
* subgroups: per property, noticing and preference in low vs high criticality scenes, in
  clips from the participant's own country vs elsewhere, and (for colour-dependent
  properties) with vs without a colour vision deficiency, with Fisher's exact test.
  With ``colour_properties_without_cvd`` the main results of colour-dependent properties
  leave out participants who report a colour vision deficiency.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from avprops.crowd import mask_ips

RATING_KEYS = ("understanding", "safety", "load")


def ended(rec: dict, config: dict) -> bool:
    """Whether the session reached an end: the last page, or the attention checks stopped it
    (``attention_fail_at`` wrong answers). A session left halfway, e.g. by reloading the page,
    did not end."""
    wrong = (rec.get("attention") or {}).get("wrong") or 0
    return bool(rec.get("finished")) or wrong >= config.get("attention_fail_at", math.inf)


def expected_trials(config: dict, n_items: int) -> tuple[int, int]:
    """BWS trials and A/B pairs (repeats and the display vs no display pair included) that every
    participant gets, as design.build_design and the experiment make them."""
    k = min(config["bws_items_per_trial"], n_items)
    bws = math.ceil(n_items * config["bws_appearances_per_item"] / k)
    pairs = config["ab_pairs_per_participant"]
    pairs = n_items if pairs in (0, None) else min(pairs, n_items)
    repeats = min(int(config.get("ab_repeats", 0)), pairs)
    return bws, pairs + repeats + int(bool(config.get("ab_no_display", False)))


def missing_data(rec: dict, expected: tuple[int, int]) -> str:
    """What a record lacks of a whole session ("" when nothing): consent, BWS trials, A/B pairs.
    The server stores whatever rows it is sent, so a record can hold a last row and little else."""
    bws, ab = expected
    missing = [] if rec.get("consent") else ["no consent"]
    if len(rec.get("bws", [])) < bws:
        missing.append(f"{len(rec.get('bws', []))} of {bws} BWS trials")
    if len(rec.get("ab", [])) < ab:
        missing.append(f"{len(rec.get('ab', []))} of {ab} A/B pairs")
    return ", ".join(missing)


def apply_exclusions(records: list[dict], config: dict,
                     n_items: int | None = None) -> tuple[list[dict], pd.DataFrame]:
    """Return the kept records and a table of every participant with the reason excluded.
    With ``n_items`` (the number of properties) a finished record must also hold every trial."""
    kept, rows = [], []
    # a later participant from an address whose earlier session ended took part twice (or shares
    # a device): keep the first one only. A session left halfway does not count, so a participant
    # who reloads the page (and so gets a new code) keeps the session they finish; debug sessions
    # do not count either
    later_same_ip, seen = set(), set()
    for rec in sorted(records, key=lambda r: r.get("arrival", 0)):
        if rec.get("ip") and not rec.get("debug"):
            if rec["ip"] in seen:
                later_same_ip.add(id(rec))
            if ended(rec, config):
                seen.add(rec["ip"])
    expected = expected_trials(config, n_items) if n_items else None
    versions = config.get("design_versions") or []
    for rec in records:
        reasons = []
        if config["exclude_debug"] and rec.get("debug"):
            reasons.append("debug session")
        if rec.get("study_id") and rec["study_id"] != config["study_id"]:
            reasons.append(f"other study ({rec['study_id']})")
        if versions and rec.get("design_generated_at") not in versions:
            reasons.append(f"other design version ({rec.get('design_generated_at')})")
        if config.get("exclude_duplicate_ip", True) and id(rec) in later_same_ip:
            reasons.append("same IP address as an earlier participant")
        if config["exclude_unfinished"] and rec.get("finished") is False:
            reasons.append("did not finish")
        if expected and config.get("exclude_incomplete", True) and rec.get("finished"):
            missing = missing_data(rec, expected)
            if missing:
                reasons.append(f"incomplete data ({missing})")
        attention = rec.get("attention") or {}
        if not attention.get("passed", False):
            reasons.append(f"more than {config['attention_max_wrong']} wrong attention answers")
        rts = [t["rt_ms"] for t in rec.get("bws", [])]
        if rts and float(np.median(rts)) < config["min_median_bws_rt_ms"]:
            reasons.append("BWS answered too fast")
        ab = rec.get("ab", [])
        if ab and np.mean([bool(p.get("hidden")) for p in ab]) > config["max_hidden_ab_share"]:
            reasons.append("tab hidden during too many A/B pairs")
        consistency = repeat_consistency(rec)
        if config.get("min_repeat_consistency", 0) > 0 and consistency is not None \
                and consistency < config["min_repeat_consistency"]:
            reasons.append(f"repeated pairs answered consistently in {consistency:.0%}")
        rows.append({"pid": rec.get("pid"), "excluded": bool(reasons), "reasons": "; ".join(reasons)})
        if not reasons:
            kept.append(rec)
    return kept, pd.DataFrame(rows, columns=["pid", "excluded", "reasons"])


def first_pairs(rec: dict) -> list[dict]:
    """The A/B pairs of the properties, without repeats and the no-display pair."""
    return [p for p in rec.get("ab", []) if not p.get("repeat") and p["property"] != "no_display"]


def repeat_consistency(rec: dict) -> float | None:
    """Share of repeated pairs where the same level was preferred both times."""
    first = {p["property"]: p for p in first_pairs(rec)}
    same = [p["preferred_level"] == first[p["property"]]["preferred_level"]
            for p in rec.get("ab", []) if p.get("repeat") and p["property"] in first]
    return float(np.mean(same)) if same else None


def prefers_no_display(rec: dict) -> bool | None:
    pair = next((p for p in rec.get("ab", []) if p["property"] == "no_display"), None)
    return None if pair is None else pair["preferred_level"] == "none"


def questionnaire_scores(rec: dict, questionnaires: list[dict]) -> dict[str, float]:
    """Mean item score per questionnaire, reversed items flipped; NaN if incomplete."""
    scores = {}
    for q in questionnaires:
        answers = rec.get("questionnaires", {}).get(q["id"], {})
        n = len(q["labels"])
        values = [(n + 1 - answers[i["id"]]) if i["reversed"] else answers[i["id"]]
                  for i in q["items"] if i["id"] in answers]
        scores[q["id"]] = float(np.mean(values)) if len(values) == len(q["items"]) else np.nan
    return scores


def bws_individual(records: list[dict], prop_ids: list[str]) -> pd.DataFrame:
    """Participants x properties matrix of (best - worst) / shown, NaN where never shown."""
    rows = {}
    for rec in records:
        shown = dict.fromkeys(prop_ids, 0)
        score = dict.fromkeys(prop_ids, 0)
        for trial in rec.get("bws", []):
            for item in trial["items"]:
                if item in shown:
                    shown[item] += 1
            if trial["best"] in score:
                score[trial["best"]] += 1
            if trial["worst"] in score:
                score[trial["worst"]] -= 1
        rows[rec["pid"]] = {p: (score[p] / shown[p] if shown[p] else np.nan) for p in prop_ids}
    return pd.DataFrame.from_dict(rows, orient="index", columns=prop_ids)


def bws_summary(individual: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = individual.to_numpy(dtype=float)
    n = len(values)
    means = np.nanmean(values, axis=0) if n else np.full(values.shape[1], np.nan)
    if n > 1:
        boots = np.array([np.nanmean(values[rng.integers(0, n, n)], axis=0) for _ in range(n_boot)])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    else:
        lo = hi = np.full(values.shape[1], np.nan)
    out = pd.DataFrame({
        "bws_score": means,
        "bws_ci_low": lo,
        "bws_ci_high": hi,
        "bws_n": np.sum(~np.isnan(values), axis=0),
    }, index=individual.columns)
    out["bws_rank"] = out["bws_score"].rank(ascending=False, method="min")
    return out


def _wilcoxon_p(diffs: list[float]) -> float:
    diffs = [d for d in diffs if d != 0]
    if len(diffs) < 6:
        return np.nan
    return float(stats.wilcoxon(diffs).pvalue)


def has_cvd(rec: dict) -> bool:
    return (rec.get("demographics") or {}).get("colour_vision_deficiency") == "Yes"


def ab_summary(records: list[dict], prop_ids: list[str], skip_cvd_for: set[str] | None = None) -> pd.DataFrame:
    """A/B results per property. Participants who report a colour vision deficiency are
    left out of the properties in ``skip_cvd_for``."""
    skip_cvd_for = skip_cvd_for or set()
    by_prop: dict[str, list[dict]] = {p: [] for p in prop_ids}
    for rec in records:
        for pair in first_pairs(rec):
            if pair["property"] in by_prop and not (pair["property"] in skip_cvd_for and has_cvd(rec)):
                by_prop[pair["property"]].append(pair)

    rows = {}
    for prop, pairs in by_prop.items():
        noticed = [p for p in pairs if p["noticed"]]
        prefer_high = [p["preferred_level"] == "high" for p in noticed]
        share_high = float(np.mean(prefer_high)) if prefer_high else np.nan
        row = {
            "ab_n": len(pairs),
            "noticed_share": float(np.mean([p["noticed"] for p in pairs])) if pairs else np.nan,
            "pref_n": len(noticed),
            "prefer_high_share": share_high,
            "majority_share": max(share_high, 1 - share_high) if prefer_high else np.nan,
            "majority_level": ("high" if share_high >= 0.5 else "low") if prefer_high else "",
            "importance_mean": float(np.mean([p["importance"] for p in pairs])) if pairs else np.nan,
        }
        for key in RATING_KEYS:
            diffs = [p["ratings"]["high"][key] - p["ratings"]["low"][key]
                     for p in pairs if p.get("ratings")]
            row[f"{key}_diff_mean"] = float(np.mean(diffs)) if diffs else np.nan
            row[f"{key}_diff_p"] = _wilcoxon_p(diffs)
        rows[prop] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def comprehension_summary(records: list[dict], prop_ids: list[str]) -> pd.DataFrame:
    rows = {}
    for prop in prop_ids:
        answers = [c for rec in records for c in rec.get("comprehension", []) if c["property"] == prop]
        low = [bool(c["correct"]) for c in answers if c["level"] == "low"]
        high = [bool(c["correct"]) for c in answers if c["level"] == "high"]
        p = np.nan
        if low and high:
            table = [[sum(high), len(high) - sum(high)], [sum(low), len(low) - sum(low)]]
            p = float(stats.fisher_exact(table).pvalue)
        rows[prop] = {
            "comprehension_n_low": len(low), "comprehension_low": float(np.mean(low)) if low else np.nan,
            "comprehension_n_high": len(high), "comprehension_high": float(np.mean(high)) if high else np.nan,
            "comprehension_p": p,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def preference_by_score(records: list[dict], prop_ids: list[str], questionnaires: list[dict],
                        min_n: int = 10) -> pd.DataFrame:
    """Point-biserial r between each questionnaire score and choosing the high level
    (among participants who noticed the difference)."""
    rows = {}
    for prop in prop_ids:
        row = {}
        for q in questionnaires:
            pairs = [(questionnaire_scores(rec, [q])[q["id"]], p["preferred_level"] == "high")
                     for rec in records for p in first_pairs(rec) if p["property"] == prop and p["noticed"]]
            pairs = [(score, high) for score, high in pairs if not np.isnan(score)]
            r = p = np.nan
            chose = [high for _, high in pairs]
            if len(pairs) >= min_n and 0 < sum(chose) < len(chose) and np.std([s for s, _ in pairs]) > 0:
                result = stats.pointbiserialr(chose, [score for score, _ in pairs])
                r, p = float(result.statistic), float(result.pvalue)
            row[f"high_vs_{q['id']}_r"] = r
            row[f"high_vs_{q['id']}_p"] = p
        rows[prop] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def _group_of(factor: str, rec: dict, pair: dict, scenes: dict[str, dict]) -> str | None:
    scene = scenes.get(pair["scene"], {})
    if factor == "criticality":
        return scene.get("criticality") or None
    if factor == "familiarity":
        home = (rec.get("demographics") or {}).get("country_residence")
        if not home or not scene.get("country"):
            return None
        return "own country" if home == scene["country"] else "other country"
    if factor == "variant":
        return pair.get("high_variant") or None
    if factor == "colour_vision":
        answer = (rec.get("demographics") or {}).get("colour_vision_deficiency")
        return {"Yes": "deficiency", "No": "typical"}.get(answer)
    raise ValueError(factor)


def _group_test(table: list[list[int]]) -> float:
    """Fisher's exact test for two groups, chi-square for more (e.g. palette variants)."""
    if len(table) < 2 or any(sum(row) == 0 for row in table):
        return np.nan
    if len(table) == 2:
        return float(stats.fisher_exact(table).pvalue)
    columns = np.array(table).sum(axis=0)
    if (columns == 0).any():
        return np.nan
    return float(stats.chi2_contingency(table).pvalue)


def subgroup_summary(records: list[dict], properties: list[dict], scenes: list[dict]) -> pd.DataFrame:
    """Noticing and preference per property in two groups of a factor, with Fisher's exact
    test on choosing the high level (participants who noticed)."""
    scene_meta = {s["id"]: s for s in scenes}
    rows = []
    for prop in properties:
        factors = (["criticality", "familiarity"] + (["colour_vision"] if prop.get("colour_dependent") else [])
                   + (["variant"] if "variants" in prop.get("render", {}) else []))
        for factor in factors:
            groups: dict[str, list[dict]] = {}
            for rec in records:
                for pair in first_pairs(rec):
                    if pair["property"] == prop["id"]:
                        group = _group_of(factor, rec, pair, scene_meta)
                        if group:
                            groups.setdefault(group, []).append(pair)
            stats_by_group, seen_by_group = {}, {}
            for group, pairs in sorted(groups.items()):
                noticed = [p for p in pairs if p["noticed"]]
                high = sum(p["preferred_level"] == "high" for p in noticed)
                stats_by_group[group] = (len(noticed), high)
                seen_by_group[group] = (len(pairs), len(noticed))
                rows.append({"property": prop["id"], "factor": factor, "group": group, "n": len(pairs),
                             "noticed_share": float(np.mean([p["noticed"] for p in pairs])),
                             "pref_n": len(noticed),
                             "prefer_high_share": high / len(noticed) if noticed else np.nan})
            p_pref = p_seen = np.nan
            pref_table = [[h, n - h] for n, h in stats_by_group.values() if n > 0]
            seen_table = [[s_, t - s_] for t, s_ in seen_by_group.values()]
            p_pref = _group_test(pref_table)
            p_seen = _group_test(seen_table)
            for row in rows[len(rows) - len(stats_by_group):]:
                row["p_noticed_differs"] = p_seen
                row["p_prefer_high_differs"] = p_pref
    return pd.DataFrame(rows, columns=["property", "factor", "group", "n", "noticed_share", "p_noticed_differs",
                                       "pref_n", "prefer_high_share", "p_prefer_high_differs"])


def redundant_pairs(individual: pd.DataFrame, min_abs_r: float, min_n: int = 10) -> pd.DataFrame:
    """Property pairs whose individual BWS scores correlate strongly across participants."""
    rows = []
    cols = list(individual.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both = individual[[a, b]].dropna()
            if len(both) < min_n or both[a].std() == 0 or both[b].std() == 0:
                continue
            r = float(np.corrcoef(both[a], both[b])[0, 1])
            if abs(r) >= min_abs_r:
                rows.append({"property_a": a, "property_b": b, "r": r, "n": len(both)})
    return pd.DataFrame(rows, columns=["property_a", "property_b", "r", "n"])


def select_properties(summary: pd.DataFrame, redundant: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply the pre-registered rules; add a ``status`` column and the final ``selected`` flag."""
    summary = summary.sort_values("bws_score", ascending=False).copy()
    n_top = int(np.ceil(len(summary) * config["matters_top_fraction"]))
    top = set(summary.index[:n_top])
    alpha = 0.05

    partners: dict[str, set[str]] = {}
    for row in redundant.itertuples():
        partners.setdefault(row.property_a, set()).add(row.property_b)
        partners.setdefault(row.property_b, set()).add(row.property_a)

    status, chosen = {}, []
    for prop, row in summary.iterrows():
        rating_effect = any(row.get(f"{k}_diff_p", np.nan) < alpha for k in RATING_KEYS)
        if not row["noticed_share"] >= config["noticed_min_share"]:
            status[prop] = "not noticed"
        elif prop not in top and not rating_effect:
            status[prop] = "does not matter"
        elif pd.isna(row["majority_share"]):
            status[prop] = "no preference data"
        elif row["majority_share"] > config["disagreement_max_majority_share"]:
            status[prop] = f"consensus: fix at {row['majority_level']}"
        elif partners.get(prop, set()) & set(chosen):
            status[prop] = "redundant with " + ", ".join(sorted(partners[prop] & set(chosen)))
        elif len(chosen) < config["select_n"]:
            chosen.append(prop)
            status[prop] = "selected"
        else:
            status[prop] = "eligible, below cut-off"
    summary["status"] = pd.Series(status)
    summary["selected"] = summary["status"] == "selected"
    return summary


def analyse(records: list[dict], properties: list[dict], config: dict,
            questionnaires: list[dict] | None = None, scenes: list[dict] | None = None) -> dict[str, Any]:
    questionnaires = questionnaires or []
    prop_ids = [p["id"] for p in properties]
    kept, exclusions = apply_exclusions(records, config, len(prop_ids))
    # a design.json made again (another generated_at) can assign other trials: never pool
    # participants of different versions without saying which ones belong to the study
    versions = sorted({str(r.get("design_generated_at")) for r in kept})
    if len(versions) > 1 and not config.get("design_versions"):
        raise ValueError(f"the kept participants saw {len(versions)} different designs ({', '.join(versions)}); "
                         "list the one(s) to analyse in design_versions in the config")
    mask_ips(records)  # the addresses are not needed after the exclusions; no output holds them
    individual = bws_individual(kept, prop_ids)
    summary = bws_summary(individual, config["bootstrap_samples"], config["bootstrap_seed"])
    skip_cvd = ({p["id"] for p in properties if p.get("colour_dependent")}
                if config.get("colour_properties_without_cvd") else set())
    summary = summary.join(ab_summary(kept, prop_ids, skip_cvd))
    summary = summary.join(comprehension_summary(kept, prop_ids))
    summary = summary.join(preference_by_score(kept, prop_ids, questionnaires))
    summary.insert(0, "name", [next(p["name"] for p in properties if p["id"] == i) for i in summary.index])
    redundant = redundant_pairs(individual, config["redundancy_min_abs_r"])
    summary = select_properties(summary, redundant, config)
    participants = pd.DataFrame([{
        "pid": r["pid"], "ip": r.get("ip"), "finished": r.get("finished"), "debug": r.get("debug"),
        "attention_wrong": (r.get("attention") or {}).get("wrong"),
        "bws_trials": len(r.get("bws", [])), "ab_pairs": len(first_pairs(r)),
        "repeat_consistency": repeat_consistency(r), "prefers_no_display": prefers_no_display(r),
        "comprehension_correct": (float(np.mean([bool(c["correct"]) for c in r["comprehension"]]))
                                  if r.get("comprehension") else None),
        "country_residence": (r.get("demographics") or {}).get("country_residence"),
        "nationality": (r.get("demographics") or {}).get("nationality"),
        "colour_vision_deficiency": (r.get("demographics") or {}).get("colour_vision_deficiency"),
        **questionnaire_scores(r, questionnaires),
    } for r in records]).merge(exclusions, on="pid")
    kept_rows = participants[~participants["excluded"]]
    return {
        "n_records": len(records),
        "n_kept": len(kept),
        "participants": participants,
        "repeat_consistency": float(kept_rows["repeat_consistency"].dropna().mean())
        if kept_rows["repeat_consistency"].notna().any() else np.nan,
        "prefers_no_display": float(kept_rows["prefers_no_display"].dropna().astype(float).mean())
        if kept_rows["prefers_no_display"].notna().any() else np.nan,
        "exclusions": exclusions,
        "summary": summary,
        "subgroups": subgroup_summary(kept, properties, scenes or []),
        "redundant": redundant,
        "open_answers": pd.DataFrame(
            [{"pid": r["pid"], "baseline": (r.get("open_question") or {}).get("text", ""),
              "final_comment": r.get("final_comment", "")} for r in kept],
            columns=["pid", "baseline", "final_comment"]),
    }
