"""The anonymised dataset that is published (data management plan, question 19).

The study's codes are already random (the Microworkers ID never reaches the data; only
Microworkers can link a completion code to a worker, until payments.csv is deleted). For
publication the records are reduced further:

* every completion code is replaced by a new number (P0001, ...), in random order, so the
  published IDs cannot be matched to codes or to the order of arrival;
* IP addresses, arrival order and platform session fields are left out;
* age is published in bands;
* open answers are published from a separate file that the researcher reads and redacts by
  hand first (column ``redacted`` false until then).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

AGE_BANDS = [(18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 99)]
DEMOGRAPHICS = ("gender", "nationality", "country_residence", "licence", "car_frequency",
                "automation_experience", "av_ride", "ar_familiarity", "colour_vision_deficiency")


def age_band(age: Any) -> str:
    try:
        years = int(float(age))
    except (TypeError, ValueError):
        return ""
    return next((f"{lo}-{hi}" if hi < 99 else f"{lo}+" for lo, hi in AGE_BANDS if lo <= years <= hi), "")


def anonymise(records: list[dict], seed: int | None = None,
              exclusions: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Tables of the published dataset. ``seed`` only for tests: the real export draws the
    new IDs from fresh randomness, so they cannot be reproduced from the codes."""
    order = np.random.default_rng(seed).permutation(len(records))
    new_id = {records[i]["pid"]: f"P{k + 1:04d}" for k, i in enumerate(order)}
    reasons = ({} if exclusions is None else exclusions.set_index("pid")["reasons"].to_dict())

    participants, bws, ab, comprehension, open_answers = [], [], [], [], []
    for rec in records:
        pid = new_id[rec["pid"]]
        demo = rec.get("demographics") or {}
        excluded = reasons.get(rec["pid"])
        participants.append({"participant": pid, "finished": rec.get("finished"),
                             # whether and why the analysis left them out, so N can be reproduced
                             "excluded": None if excluded is None else bool(excluded),
                             "exclusion_reasons": excluded or "",
                             "attention_wrong": (rec.get("attention") or {}).get("wrong"),
                             "age_band": age_band(demo.get("age")),
                             **{k: demo.get(k) for k in DEMOGRAPHICS},
                             **{f"{q}_{item}": v for q, items in (rec.get("questionnaires") or {}).items()
                                for item, v in items.items()}})
        bws += [{"participant": pid, **{k: t.get(k) for k in ("trial", "best", "worst", "rt_ms", "hidden")},
                 "items": ";".join(t.get("items") or [])} for t in rec.get("bws", [])]
        ab += [{"participant": pid, **{k: p.get(k) for k in (
            "index", "property", "scene", "a_level", "b_level", "high_variant", "catch", "noticed", "preferred_level",
            "preference", "importance", "rt_ms", "hidden")},
            **{f"rating_{side}_{key}": v for side, ratings in (p.get("ratings") or {}).items()
               if isinstance(ratings, dict) for key, v in ratings.items()}} for p in rec.get("ab", [])]
        comprehension += [{"participant": pid, **c} for c in rec.get("comprehension", [])]
        texts = [("open_question", (rec.get("open_question") or {}).get("text", "")),
                 ("final_comment", rec.get("final_comment", ""))]
        open_answers += [{"participant": pid, "question": q, "text": text, "redacted": False}
                         for q, text in texts if text and text.strip()]
    sort = lambda rows: pd.DataFrame(rows).sort_values("participant", kind="stable") if rows else pd.DataFrame()  # noqa: E731
    return {"participants": sort(participants), "bws": sort(bws), "ab": sort(ab),
            "comprehension": sort(comprehension), "open_answers": sort(open_answers)}


def _value(v: Any) -> Any:
    """A CSV cell as the analysis expects it: NaN -> None, numpy scalars -> Python."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v.item() if hasattr(v, "item") else v


def load_public(folder, questionnaire_ids: list[str] | None = None, kept_only: bool = True) -> list[dict]:
    """Participant records rebuilt from the published CSV files (anonymise() writes them), so the
    analysis can be rerun on the public data and N reproduced. With ``kept_only`` the participants
    the original analysis excluded are left out (their flags are in participants.csv)."""
    from pathlib import Path
    folder = Path(folder)
    read = lambda name: pd.read_csv(folder / f"{name}.csv") if (folder / f"{name}.csv").exists() \
        else pd.DataFrame(columns=["participant"])  # noqa: E731
    people, bws, ab, comprehension = read("participants"), read("bws"), read("ab"), read("comprehension")
    questionnaire_ids = sorted(questionnaire_ids or [], key=len, reverse=True)
    rows_of = {name: {pid: rows for pid, rows in table.groupby("participant")}
               for name, table in (("bws", bws), ("ab", ab), ("comprehension", comprehension))}
    records = []
    for person in people.to_dict("records"):
        if kept_only and bool(person.get("excluded")):
            continue
        pid = person["participant"]
        questionnaires: dict[str, dict[str, int]] = {}
        for column, value in person.items():
            q = next((q for q in questionnaire_ids if column.startswith(q + "_")), None)
            if q and _value(value) is not None:
                questionnaires.setdefault(q, {})[column[len(q) + 1:]] = int(value)
        wrong = int(_value(person.get("attention_wrong")) or 0)
        empty = pd.DataFrame()
        bws_rows = rows_of["bws"].get(pid, empty)
        records.append({
            "pid": pid, "debug": False, "consent": True, "finished": bool(person.get("finished")),
            "demographics": {k: _value(person.get(k)) for k in DEMOGRAPHICS},
            "questionnaires": questionnaires,
            "attention": {"wrong": wrong, "passed": True},
            "bws": [{"trial": int(r["trial"]), "items": str(r["items"]).split(";"), "best": r["best"],
                     "worst": r["worst"], "rt_ms": _value(r["rt_ms"]), "hidden": bool(r["hidden"])}
                    for r in bws_rows.to_dict("records")],
            "ab": [{key: _value(r.get(key)) for key in (
                        "index", "property", "scene", "a_level", "b_level", "high_variant", "noticed",
                        "preferred_level", "preference", "importance", "rt_ms")}
                   | {"catch": bool(r.get("catch")), "hidden": bool(r.get("hidden")), "noticed": bool(r.get("noticed"))}
                   for r in rows_of["ab"].get(pid, empty).to_dict("records")],
            "comprehension": [{key: _value(r.get(key)) for key in (
                                   "scene", "property", "level", "high_variant", "answer", "correct", "rt_ms", "hidden")}
                              for r in rows_of["comprehension"].get(pid, empty).to_dict("records")],
        })
    return records
