"""List every completion code with the payment it earns, to approve submissions on Microworkers.

    uv run python payments.py --data /srv/study-data/entries.jsonl

Participants paste the code from the finish page into Microworkers as proof. A code earns:
    full     the participant reached the end of the study
    partial  the study was ended by the attention checks (the finish_fail page shows the code)
    none     the code is not in the data, or the participant left before the end
The ``review`` column repeats the analysis's exclusion reasons (answered too fast, same IP
address as an earlier participant, incomplete data, ...): the researcher decides about these
by hand, so no payment is refused automatically. "incomplete data" on a ``full`` code means the
record lacks trials: rows lost on the way, or rows sent to the server by hand to fake a
completion. Writes payments.csv to the output folder (it holds the completion codes, so keep
it with the study data, not in git).
"""

from __future__ import annotations

import argparse

import pandas as pd

from avprops.analysis import apply_exclusions
from avprops.common import load_config, load_properties, resolve
from avprops.crowd import read_rows, rows_to_records


def payment_status(rec: dict) -> str:
    if rec.get("finished"):
        return "full"
    if not (rec.get("attention") or {}).get("passed", True):
        return "partial"
    return "none"


def payments(records: list[dict], config: dict, properties: list[dict] | None = None) -> pd.DataFrame:
    """``properties`` (the screened-in ones) adds the "incomplete data" flag."""
    _, exclusions = apply_exclusions(records, config, properties)
    reasons = exclusions.set_index("pid")["reasons"]
    # the payment does not depend on these: they are flags for the researcher's own check
    review = {pid: "; ".join(r for r in text.split("; ") if r and r != "did not finish"
                             and not r.startswith("more than")) for pid, text in reasons.items()}
    return pd.DataFrame([{
        "code": rec["pid"], "platform": rec.get("platform", ""), "status": payment_status(rec),
        "attention_wrong": (rec.get("attention") or {}).get("wrong"), "debug": rec.get("debug"),
        "review": review.get(rec["pid"], ""),
    } for rec in sorted(records, key=lambda r: r.get("arrival", 0))])


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="data files")
    parser.add_argument("--out", default=config["output_dir"], help="output folder")
    parser.add_argument("--check", nargs="*", default=[], help="codes submitted on Microworkers: print what each earns")
    args = parser.parse_args()

    records = rows_to_records(read_rows([resolve(f) for f in args.data]), config["attention_max_wrong"])
    table = payments(records, config, load_properties(config))
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "payments.csv", index=False)
    real = table[~table["debug"].astype(bool)]
    print(f"{len(real)} codes: {(real.status == 'full').sum()} full, {(real.status == 'partial').sum()} partial, "
          f"{(real.status == 'none').sum()} none; {(real.review != '').sum()} to review -> {out / 'payments.csv'}")
    by_code = table.set_index("code")
    for code in args.check:
        row = by_code.loc[code] if code in by_code.index else None
        print(f"  {code}: " + ("not in the data (none)" if row is None
                               else f"{row.status}" + (f", review: {row.review}" if row.review else "")))


if __name__ == "__main__":
    main()
