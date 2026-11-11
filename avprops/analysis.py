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
3. Task B (A/B pairs): each participant compares some properties, each in two scenes, with a
   graded answer (A clearly .. no preference .. B clearly). Per property: the share who noticed
   the difference minus the false-alarm rate (noticing in catch pairs of two identical clips);
   the preference towards the high level; mean importance; and the heterogeneity H, the
   intraclass correlation of each participant's two answers: 0 when people do not care (their
   answers in the two scenes agree no more than chance), 1 when they hold stable, opposite
   preferences (review by Mark Colley, 2026-11-09).
4. Selection: only continuous and ordinal properties can be tuned, so only they are eligible;
   categorical ones are settled by the majority preference. Among the eligible properties that
   are noticed (``noticed_min_share`` after the false-alarm correction) and on which people
   differ (H at least ``heterogeneity_min``), the score importance x H ranks them (importance = the BWS score on 0..1) and the top ``select_n`` are selected. A
   bootstrap over participants gives each property's probability of being selected.

Alongside the selection:
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

from avprops.common import TUNABLE
from avprops.crowd import mask_ips

RATING_KEYS = ("understanding", "safety", "load")


def ended(rec: dict, config: dict) -> bool:
    """Whether the session reached an end: the last page, or the attention checks stopped it
    (``attention_fail_at`` wrong answers). A session left halfway, e.g. by reloading the page,
    did not end."""
    wrong = (rec.get("attention") or {}).get("wrong") or 0
    return bool(rec.get("finished")) or wrong >= config.get("attention_fail_at", math.inf)


def expected_trials(config: dict, properties: list[dict]) -> tuple[int, int]:
    """BWS trials and A/B pairs (catch pairs and the display vs no display pair included) that
    every participant gets, as design.build_design and the experiment make them."""
    from avprops.design import ab_counts
    n_items = len(properties)
    k = min(config["bws_items_per_trial"], n_items)
    bws = math.ceil(n_items * config["bws_appearances_per_item"] / k)
    tunable, categorical = ab_counts(config, properties)
    pairs = tunable * int(config["ab_scenes_per_property"]) + categorical
    return bws, pairs + int(config.get("ab_catch_pairs", 0)) + int(bool(config.get("ab_no_display", False)))


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
                     properties: list[dict] | None = None) -> tuple[list[dict], pd.DataFrame]:
    """Return the kept records and a table of every participant with the reason excluded.
    With ``properties`` a finished record must also hold every trial."""
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
    expected = expected_trials(config, properties) if properties else None
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
        false_alarms, n_catch = catch_noticed(rec)
        if config.get("max_catch_noticed") is not None and n_catch and false_alarms > config["max_catch_noticed"]:
            reasons.append(f"noticed a difference in {false_alarms} of {n_catch} identical pairs")
        rows.append({"pid": rec.get("pid"), "excluded": bool(reasons), "reasons": "; ".join(reasons)})
        if not reasons:
            kept.append(rec)
    return kept, pd.DataFrame(rows, columns=["pid", "excluded", "reasons"])


def property_pairs(rec: dict) -> list[dict]:
    """The A/B pairs of the properties, without the catch pairs and the no-display pair."""
    return [p for p in rec.get("ab", []) if not p.get("catch") and p["property"] != "no_display"]


def catch_noticed(rec: dict) -> tuple[int, int]:
    """(catch pairs in which a difference was reported, catch pairs shown)."""
    catches = [p for p in rec.get("ab", []) if p.get("catch")]
    return sum(bool(p["noticed"]) for p in catches), len(catches)


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


def icc_two(first: np.ndarray, second: np.ndarray) -> float:
    """One-way intraclass correlation of two answers per participant: the share of the
    answers' variance that lies between participants. 0 when a participant's two answers
    agree no more than chance (indifference, noise), 1 when everyone answers consistently
    but people differ (stable, opposite preferences). For a binary choice it equals the
    heterogeneity H of the review; NaN with fewer than 5 participants."""
    n = len(first)
    if n < 5:
        return np.nan
    means = (first + second) / 2
    ms_between = 2 * np.sum((means - means.mean()) ** 2) / (n - 1)
    ms_within = np.sum((first - means) ** 2 + (second - means) ** 2) / n
    total = ms_between + ms_within
    return float((ms_between - ms_within) / total) if total > 0 else 0.0


def false_alarm_rate(records: list[dict]) -> float:
    """Share of catch pairs (two identical clips) in which a difference was reported."""
    counts = [catch_noticed(rec) for rec in records]
    shown = sum(n for _, n in counts)
    return sum(k for k, _ in counts) / shown if shown else 0.0


def _pairs_by_property(records: list[dict], prop_ids: list[str],
                       skip_cvd_for: set[str]) -> dict[str, list[tuple[int, dict]]]:
    """(participant index, pair) per property; colour vision deficiency left out where asked."""
    by_prop: dict[str, list[tuple[int, dict]]] = {p: [] for p in prop_ids}
    for i, rec in enumerate(records):
        for pair in property_pairs(rec):
            if pair["property"] in by_prop and not (pair["property"] in skip_cvd_for and has_cvd(rec)):
                by_prop[pair["property"]].append((i, pair))
    return by_prop


def _two_answers(entries: list[tuple[int, dict]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Participants who chose a level in both scenes of a property: their index and the two
    choices (1 the high level, -1 the low level). Only the choice counts, as in the review's H
    for a binary choice: people who all prefer the same level, some more strongly than others,
    agree and have nothing to personalise; "no preference" answers count in the preference
    shares but not here."""
    answers: dict[int, list[float]] = {}
    for i, pair in entries:
        if pair.get("preference") is not None:
            answers.setdefault(i, []).append(float(np.sign(pair["preference"])))
    both = [(i, a[0], a[1]) for i, a in answers.items() if len(a) >= 2 and a[0] != 0 and a[1] != 0]
    if not both:
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    idx, first, second = zip(*both)
    return np.array(idx), np.array(first), np.array(second)


def ab_summary(records: list[dict], prop_ids: list[str], skip_cvd_for: set[str] | None = None) -> pd.DataFrame:
    """A/B results per property. Participants who report a colour vision deficiency are
    left out of the properties in ``skip_cvd_for``."""
    by_prop = _pairs_by_property(records, prop_ids, skip_cvd_for or set())
    false_alarms = false_alarm_rate(records)
    rows = {}
    for prop, entries in by_prop.items():
        pairs = [pair for _, pair in entries]
        levels = [p["preferred_level"] for p in pairs]
        chosen = [lv for lv in levels if lv in ("high", "low")]
        share_high = chosen.count("high") / len(chosen) if chosen else np.nan
        noticed = float(np.mean([bool(p["noticed"]) for p in pairs])) if pairs else np.nan
        _, first, second = _two_answers(entries)
        row = {
            "ab_n": len(pairs),
            "noticed_share": noticed,
            "noticed_corrected": noticed - false_alarms if pairs else np.nan,
            "prefer_high_share": levels.count("high") / len(levels) if levels else np.nan,
            "prefer_low_share": levels.count("low") / len(levels) if levels else np.nan,
            "no_preference_share": levels.count("none") / len(levels) if levels else np.nan,
            "majority_level": ("high" if share_high >= 0.5 else "low") if chosen else "",
            "majority_share": max(share_high, 1 - share_high) if chosen else np.nan,
            "preference_mean": float(np.mean([p["preference"] for p in pairs if p.get("preference") is not None]))
            if pairs else np.nan,
            "importance_mean": float(np.mean([p["importance"] for p in pairs])) if pairs else np.nan,
            "H": icc_two(first, second),
            "H_n": len(first),
        }
        for key in RATING_KEYS:
            diffs = [p["ratings"]["high"][key] - p["ratings"]["low"][key] for p in pairs if p.get("ratings")]
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
                     for rec in records for p in property_pairs(rec)
                     if p["property"] == prop and p["noticed"] and p["preferred_level"] in ("high", "low")]
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
                for pair in property_pairs(rec):
                    if pair["property"] == prop["id"]:
                        group = _group_of(factor, rec, pair, scene_meta)
                        if group:
                            groups.setdefault(group, []).append(pair)
            stats_by_group, seen_by_group = {}, {}
            for group, pairs in sorted(groups.items()):
                noticed = [p for p in pairs if p["noticed"] and p["preferred_level"] in ("high", "low")]
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


def _selection(bws: pd.Series, noticed: pd.Series, heterogeneity: pd.Series, tunable: set[str],
               config: dict) -> tuple[list[str], pd.Series]:
    """The selected properties and every property's score (importance x H; NaN if ineligible)."""
    importance = (bws.clip(-1, 1) + 1) / 2
    score = importance * heterogeneity.clip(lower=0).fillna(0)
    eligible = [p for p in bws.index if p in tunable and noticed.get(p, np.nan) >= config["noticed_min_share"]
                and heterogeneity.get(p, np.nan) >= config.get("heterogeneity_min", 0.0)]
    score = score.where(score.index.isin(eligible))
    chosen = list(score.dropna().sort_values(ascending=False).index[:config["select_n"]])
    return chosen, score


def selection_probability(records: list[dict], individual: pd.DataFrame, prop_ids: list[str],
                          tunable: set[str], skip_cvd_for: set[str], config: dict) -> pd.Series:
    """Share of bootstrap samples of participants in which each property is selected."""
    n_boot = int(config.get("selection_bootstrap", 0))
    if not n_boot or len(records) < 2:
        return pd.Series(np.nan, index=prop_ids)
    n = len(records)
    bws = individual.reindex([r["pid"] for r in records])[prop_ids].to_numpy()
    by_prop = _pairs_by_property(records, prop_ids, skip_cvd_for)
    noticed_k, noticed_n = np.zeros((n, len(prop_ids))), np.zeros((n, len(prop_ids)))
    for j, prop in enumerate(prop_ids):
        for i, pair in by_prop[prop]:
            noticed_n[i, j] += 1
            noticed_k[i, j] += bool(pair["noticed"])
    answers = {prop: _two_answers(by_prop[prop]) for prop in prop_ids}
    catch = np.array([catch_noticed(rec) for rec in records], float)
    rng = np.random.default_rng(config.get("bootstrap_seed", 0))
    counts = pd.Series(0, index=prop_ids)
    for _ in range(n_boot):
        weight = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        shown = (~np.isnan(bws)) * weight[:, None]
        bws_mean = pd.Series(np.nansum(np.nan_to_num(bws) * weight[:, None], axis=0) / np.maximum(shown.sum(0), 1),
                             index=prop_ids)
        false_alarms = (weight @ catch[:, 0]) / (weight @ catch[:, 1]) if weight @ catch[:, 1] else 0.0
        noticed = pd.Series((weight @ noticed_k) / np.maximum(weight @ noticed_n, 1) - false_alarms, index=prop_ids)
        heterogeneity = {}
        for prop, (idx, first, second) in answers.items():
            reps = weight[idx].astype(int) if len(idx) else np.zeros(0, int)
            heterogeneity[prop] = icc_two(np.repeat(first, reps), np.repeat(second, reps))
        chosen, _ = _selection(bws_mean, noticed, pd.Series(heterogeneity), tunable, config)
        counts[chosen] += 1
    return counts / n_boot


def select_properties(summary: pd.DataFrame, properties: list[dict], config: dict) -> pd.DataFrame:
    """Apply the pre-registered rule; add ``eligible``, ``score``, ``status`` and ``selected``."""
    kind = {p["id"]: p.get("type") for p in properties}
    tunable = {p for p, k in kind.items() if k in TUNABLE}
    chosen, score = _selection(summary["bws_score"], summary["noticed_corrected"], summary["H"], tunable, config)
    summary = summary.copy()
    summary["type"] = pd.Series(kind)
    summary["eligible"] = summary.index.isin(score.dropna().index)
    summary["score"] = score
    status = {}
    for prop, row in summary.iterrows():
        if prop in chosen:
            status[prop] = "selected"
        elif not row["noticed_corrected"] >= config["noticed_min_share"]:
            status[prop] = "not noticed"
        elif prop not in tunable:
            status[prop] = f"categorical: settle at {row['majority_level']}" if row["majority_level"] \
                else "categorical: no preference data"
        elif not row["H"] >= config.get("heterogeneity_min", 0.0):
            status[prop] = f"people agree: fix at {row['majority_level']}" if row["majority_level"] \
                else "people agree: no preference data"
        else:
            status[prop] = "eligible, below cut-off"
    summary["status"] = pd.Series(status)
    summary["selected"] = summary["status"] == "selected"
    return summary.sort_values(["selected", "score", "bws_score"], ascending=False)


def analyse(records: list[dict], properties: list[dict], config: dict,
            questionnaires: list[dict] | None = None, scenes: list[dict] | None = None) -> dict[str, Any]:
    questionnaires = questionnaires or []
    prop_ids = [p["id"] for p in properties]
    kept, exclusions = apply_exclusions(records, config, properties)
    # a design.json made again (another generated_at) can assign other trials: never pool
    # participants of different versions without saying which ones belong to the study
    versions = sorted({str(r.get("design_generated_at")) for r in kept})
    if len(versions) > 1 and not config.get("design_versions"):
        raise ValueError(f"the kept participants saw {len(versions)} different designs ({', '.join(versions)}); "
                         "list the one(s) to analyse in design_versions in the config")
    mask_ips(records)  # the addresses are not needed after the exclusions; no output holds them
    individual = bws_individual(kept, prop_ids)
    summary = bws_summary(individual, config["bootstrap_samples"], config["bootstrap_seed"])
    skip_cvd = ({p["id"] for p in properties if p.get("colour_dependent") and not p.get("keep_colour_vision_deficiency")}
                if config.get("colour_properties_without_cvd") else set())
    summary = summary.join(ab_summary(kept, prop_ids, skip_cvd))
    summary = summary.join(comprehension_summary(kept, prop_ids))
    summary = summary.join(preference_by_score(kept, prop_ids, questionnaires))
    summary.insert(0, "name", [next(p["name"] for p in properties if p["id"] == i) for i in summary.index])
    summary = select_properties(summary, properties, config)
    tunable = {p["id"] for p in properties if p.get("type") in TUNABLE}
    summary["p_selected"] = selection_probability(kept, individual, prop_ids, tunable, skip_cvd, config)
    participants = pd.DataFrame([{
        "pid": r["pid"], "ip": r.get("ip"), "design": r.get("design_generated_at"),
        "finished": r.get("finished"), "debug": r.get("debug"),
        "attention_wrong": (r.get("attention") or {}).get("wrong"),
        "bws_trials": len(r.get("bws", [])), "ab_pairs": len(property_pairs(r)),
        "catch_noticed": catch_noticed(r)[0], "catch_pairs": catch_noticed(r)[1],
        "prefers_no_display": prefers_no_display(r),
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
        "false_alarm_rate": false_alarm_rate(kept),
        "prefers_no_display": float(kept_rows["prefers_no_display"].dropna().astype(float).mean())
        if kept_rows["prefers_no_display"].notna().any() else np.nan,
        "exclusions": exclusions,
        "summary": summary,
        "subgroups": subgroup_summary(kept, properties, scenes or []),
        "open_answers": pd.DataFrame(
            [{"pid": r["pid"], "baseline": (r.get("open_question") or {}).get("text", ""),
              "final_comment": r.get("final_comment", "")} for r in kept],
            columns=["pid", "baseline", "final_comment"]),
    }
