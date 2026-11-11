"""Write the anonymised dataset for publication (e.g. on 4TU.ResearchData).

    uv run python export_public.py --data /srv/study-data/entries.jsonl --out _output/public

Writes participants.csv, bws.csv, ab.csv, comprehension.csv and open_answers.csv (see
avprops/publish.py for what is removed). Before publishing, read open_answers.csv, remove
anything that could identify a person and set ``redacted`` to true on every row.
"""

from __future__ import annotations

import argparse

from avprops.analysis import apply_exclusions
from avprops.common import load_config, load_properties, resolve
from avprops.crowd import read_rows, rows_to_records
from avprops.publish import anonymise


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="data files")
    parser.add_argument("--out", default="_output/public", help="output folder")
    parser.add_argument("--include-debug", action="store_true", help="keep debug sessions (default: left out)")
    args = parser.parse_args()

    records = rows_to_records(read_rows([resolve(f) for f in args.data]), config["attention_max_wrong"])
    if not args.include_debug:
        records = [r for r in records if not r.get("debug")]
    _, exclusions = apply_exclusions(records, config, load_properties(config))
    tables = anonymise(records, exclusions=exclusions)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    print(f"{len(records)} participants -> {out}; redact open_answers.csv by hand before publishing")


if __name__ == "__main__":
    main()
