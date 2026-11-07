"""Read the crowdsourced data and turn it into one record per participant.

The experiment sends batches of jsPsych rows to the server, which stores each batch as
one document ``{"data": [rows]}`` in the MongoDB collection ``entries`` (or, without a
database, as one line of data/entries.jsonl). A ``mongoexport`` of the collection has
the same shape, one document per line, or a JSON array with ``--jsonArray``.

Rows are grouped by ``worker_code``; a row sent twice is kept once. Every row carries a
``task`` field set in public/js/experiment.js, which decides how it is read.

Each document also holds the participant's IP address (``ip``), added by the server. It is
only used to find people who took part more than once (analysis.apply_exclusions keeps the
first participant of each address), after which ``mask_ips`` replaces the addresses with
placeholders (0.0.0.1, 0.0.0.2, ...), as in bazilinskyy/text-crowdsourced, so that no
output holds them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


def read_documents(path: Path) -> list[dict]:
    """Documents of a mongoexport (lines or a JSON array) or of data/entries.jsonl."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_rows(paths: Iterable[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for doc in read_documents(path):
            data = doc.get("data", [])
            ip = doc.get("ip")
            for row in data if isinstance(data, list) else [data]:
                rows.append(dict(row, ip=ip) if ip and "ip" not in row else row)
    return rows


def mask_ips(records: list[dict]) -> None:
    """Replace each IP address by a placeholder, the same one for the same address, numbered
    in order of arrival."""
    masked: dict[str, str] = {}
    for rec in sorted(records, key=lambda r: r.get("arrival", 0)):
        if rec.get("ip"):
            rec["ip"] = masked.setdefault(rec["ip"], f"0.0.0.{len(masked) + 1}")


def download(uri: str, out: Path) -> None:
    """Export the ``entries`` collection with mongoexport (as in ehmi-vehicles)."""
    if not uri:
        raise SystemExit("download_data is on but mongodb_uri is empty in secret")
    if shutil.which("mongoexport") is None:
        raise SystemExit("mongoexport is not installed (MongoDB Database Tools)")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["mongoexport", "--uri", uri, "--collection", "entries", "--type", "json", "--out", str(out)],
                   check=True)


def _rt(row: dict) -> int | None:
    return None if row.get("rt") is None else int(row["rt"])


def rows_to_records(rows: list[dict], attention_max_wrong: int) -> list[dict[str, Any]]:
    """Group jsPsych rows by participant into the record format the analysis reads."""
    by_worker: dict[str, dict[int, dict]] = {}
    arrival: dict[str, int] = {}  # order in which participants' first rows reached the server
    for row in rows:
        code = row.get("worker_code")
        if code is None or row.get("trial_index") is None:
            continue
        by_worker.setdefault(code, {})[int(row["trial_index"])] = row
        arrival.setdefault(code, len(arrival))

    records = []
    for code, indexed in by_worker.items():
        trials = [indexed[i] for i in sorted(indexed)]
        first = trials[0]
        record: dict[str, Any] = {
            "pid": code,
            "arrival": arrival[code],
            "ip": next((t["ip"] for t in trials if t.get("ip")), None),
            "study_id": first.get("study_id"),
            "design_generated_at": first.get("design_generated_at"),
            "debug": any(bool(t.get("debug")) for t in trials),
            "platform": first.get("platform", ""),
            "prolific_study_id": first.get("prolific_study_id", ""),
            "prolific_session_id": first.get("prolific_session_id", ""),
            "consent": False,
            "demographics": {},
            "open_question": None,
            "questionnaires": {},
            "comprehension": [],
            "bws": [],
            "ab": [],
            "final_comment": "",
            "finished": False,
        }
        attention = []
        for t in trials:
            task = t.get("task")
            response = t.get("response") or {}
            if task == "instructions" and "bws_version" in t:
                record["bws_version"] = t["bws_version"]
            elif task == "consent":
                record["consent"] = response.get("consent") == "Yes"
            elif task == "demographics":
                record["demographics"].update(response)
            elif task == "open_question":
                record["open_question"] = {"scene": t.get("scene"), "text": response.get("open", ""),
                                           "rt_ms": _rt(t), "hidden": bool(t.get("hidden"))}
            elif task == "bws":
                record["bws"].append({"trial": t["trial"], "items": t["items"], "best": t["best"],
                                      "worst": t["worst"], "rt_ms": _rt(t), "hidden": bool(t.get("hidden"))})
            elif task == "ab":
                record["ab"].append({key: t.get(key) for key in (
                    "index", "property", "scene", "a_level", "b_level", "noticed", "preferred_option",
                    "preferred_level", "importance", "ratings", "high_variant")}
                    | {"repeat": bool(t.get("repeat")), "rt_ms": _rt(t), "hidden": bool(t.get("hidden"))})
            elif task == "questionnaire":
                # survey-likert stores the index of the chosen label; store 1..n
                record["questionnaires"].setdefault(t["questionnaire"], {}).update(
                    {item: int(index) + 1 for item, index in response.items()})
            elif task == "comprehension":
                record["comprehension"].append({key: t.get(key) for key in (
                    "scene", "property", "level", "high_variant", "answer", "correct")}
                    | {"rt_ms": _rt(t), "hidden": bool(t.get("hidden"))})
            elif task == "attention":
                attention.append(bool(t.get("correct")))
            elif task == "final":
                record["final_comment"] = response.get("comment", "")
                record["finished"] = True
        wrong = attention.count(False)
        record["attention"] = {"shown": len(attention), "wrong": wrong, "passed": wrong <= attention_max_wrong}
        records.append(record)
    return records
