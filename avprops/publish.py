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


def anonymise(records: list[dict], seed: int | None = None) -> dict[str, pd.DataFrame]:
    """Tables of the published dataset. ``seed`` only for tests: the real export draws the
    new IDs from fresh randomness, so they cannot be reproduced from the codes."""
    order = np.random.default_rng(seed).permutation(len(records))
    new_id = {records[i]["pid"]: f"P{k + 1:04d}" for k, i in enumerate(order)}

    participants, bws, ab, comprehension, open_answers = [], [], [], [], []
    for rec in records:
        pid = new_id[rec["pid"]]
        demo = rec.get("demographics") or {}
        participants.append({"participant": pid, "finished": rec.get("finished"),
                             "attention_wrong": (rec.get("attention") or {}).get("wrong"),
                             "age_band": age_band(demo.get("age")),
                             **{k: demo.get(k) for k in DEMOGRAPHICS},
                             **{f"{q}_{item}": v for q, items in (rec.get("questionnaires") or {}).items()
                                for item, v in items.items()}})
        bws += [{"participant": pid, **{k: t.get(k) for k in ("trial", "best", "worst", "rt_ms", "hidden")},
                 "items": ";".join(t.get("items") or [])} for t in rec.get("bws", [])]
        ab += [{"participant": pid, **{k: p.get(k) for k in (
            "index", "property", "scene", "a_level", "b_level", "high_variant", "noticed", "preferred_level",
            "importance", "repeat", "rt_ms", "hidden")},
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
