"""Analyse the crowdsourced data and apply the selection rules.

    uv run python analysis.py                                   # files_heroku in path_data
    uv run python analysis.py --data data/entries.jsonl         # local runs without MongoDB
    uv run python analysis.py --data data/simulated/entries.jsonl --out _output/simulated
    uv run python analysis.py --public _output/public --out _output/from_public   # the published data

With ``download_data`` true in config, the MongoDB collection is first exported into
the first of ``files_heroku`` (needs mongoexport and ``mongodb_uri`` in secret).

Writes to the output folder:
    participants.csv   every participant: finished, attention, questionnaire scores, false
                       alarms in the catch pairs, display vs no display, comprehension,
                       excluded and why
    summary.csv        one row per property: BWS score + CI, noticing, preference, H, score,
                       status and probability of being selected
    subgroups.csv      noticing and preference per property by scene criticality, own vs
                       other country, and colour vision (colour-dependent properties)
    open_answers.csv   the open-question answers, for thematic coding
    report.md          the selected properties and where every other one stopped
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from avprops.analysis import analyse
from avprops.common import config_fingerprint, load_config, load_properties, load_questionnaires, load_scenes, load_secret, resolve
from avprops.crowd import download, read_rows, rows_to_records


def write_report(result: dict, config: dict, path: Path) -> None:
    summary = result["summary"]
    lines = [
        f"# Property selection: {config['study_id']}",
        "",
        f"Participants: {result['n_records']} records, {result['n_kept']} kept after exclusions.",
        f"Settings: {config_fingerprint(config)}; design version(s): "
        f"{', '.join(sorted(set(result['participants'].query('not excluded')['design'].dropna().astype(str)))) or 'none'}.",
        "",
        "## Selected",
        "",
    ]
    selected = summary[summary["selected"]]
    if selected.empty:
        lines.append("No property passed every rule.")
    for prop, row in selected.iterrows():
        lines.append(f"- **{row['name']}** (`{prop}`): BWS {row['bws_score']:+.2f} "
                     f"[{row['bws_ci_low']:+.2f}, {row['bws_ci_high']:+.2f}], "
                     f"noticed {row['noticed_corrected']:.0%} (corrected), H {row['H']:.2f}, "
                     f"selected in {row['p_selected']:.0%} of bootstrap samples")

    def pct(v):
        return "" if pd.isna(v) else f"{v:.0%}"

    def num(v, fmt):
        return "n/a" if pd.isna(v) else format(v, fmt)

    lines += ["", "## All properties", "",
              "| Property | Type | BWS score | Noticed (corrected) | H | Score | P(selected) | Status |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for prop, row in summary.iterrows():
        lines.append(f"| {row['name']} | {row['type']} | {row['bws_score']:+.2f} | {pct(row['noticed_corrected'])} | "
                     f"{num(row['H'], '.2f')} | {num(row['score'], '.2f')} | {pct(row['p_selected'])} | "
                     f"{row['status']} |")
    lines += ["", "## False alarms and wanting a display", "",
              f"- Catch pairs (two identical clips) in which a difference was reported: "
              f"{num(result['false_alarm_rate'], '.0%')}; subtracted from every property's noticing.",
              f"- Participants who preferred no display at all: {num(result['prefers_no_display'], '.0%')}.", ""]
    if "comprehension_n_low" in summary and (summary["comprehension_n_low"] + summary["comprehension_n_high"]).sum():
        lines += ["## Comprehension: what will the car do next?", "",
                  "Between participants: each answered one question per scene, after a clip at the low or high "
                  "level of one property. Groups are small, so only large differences can show.", "",
                  "| Property | Correct after low clip | Correct after high clip | p (Fisher) |",
                  "| --- | --- | --- | --- |"]
        for prop, row in summary.iterrows():
            if row["comprehension_n_low"] + row["comprehension_n_high"] == 0:
                continue
            lines.append(f"| {row['name']} | {num(row['comprehension_low'], '.0%')} (n={row['comprehension_n_low']:.0f}) | "
                         f"{num(row['comprehension_high'], '.0%')} (n={row['comprehension_n_high']:.0f}) | "
                         f"{num(row['comprehension_p'], '.3f')} |")
        lines.append("")
    score_cols = [c for c in summary.columns if c.startswith("high_vs_") and c.endswith("_r")]
    if score_cols:
        lines += ["## Do preferences split by user type?", "",
                  "Point-biserial r between the questionnaire score and choosing the high level "
                  "(participants who noticed the difference). The p values are not corrected: with "
                  f"{len(score_cols) * len(summary)} tests, about {0.05 * len(score_cols) * len(summary):.0f} "
                  "will fall below .05 by chance, so treat single hits with care.", "",
                  "| Property | " + " | ".join(c[len("high_vs_"):-2] for c in score_cols) + " |",
                  "| --- | " + " | ".join("---" for _ in score_cols) + " |"]
        for prop, row in summary.iterrows():
            cells = [f"{num(row[c], '+.2f')} (p={num(row[c[:-2] + '_p'], '.3f')})" for c in score_cols]
            lines.append(f"| {row['name']} | " + " | ".join(cells) + " |")
        lines.append("")
    kept_people = result["participants"].query("not excluded")
    countries = kept_people["country_residence"].dropna().value_counts()
    sub = result["subgroups"]
    lines += ["## Subgroups", "",
              f"Countries of residence (kept participants): "
              + (", ".join(f"{code} {n}" for code, n in countries.head(10).items()) or "not recorded")
              + (f", and {len(countries) - 10} more" if len(countries) > 10 else "") + ".",
              f"Colour vision deficiency reported by "
              f"{(kept_people['colour_vision_deficiency'] == 'Yes').sum()} of {len(kept_people)}"
              + ("; they are left out of the main results of colour-dependent properties."
                 if config.get("colour_properties_without_cvd") else "."), ""]
    n_tests = 2 * len(sub.drop_duplicates(["property", "factor"]))
    differs = sub[(sub["p_prefer_high_differs"] < 0.05) | (sub["p_noticed_differs"] < 0.05)]
    differs = differs.drop_duplicates(["property", "factor"])
    if differs.empty:
        lines.append("No property differed between subgroups at p < .05 (full table in subgroups.csv).")
    else:
        lines += [f"Differences between groups at p < .05 (full table in subgroups.csv). The p values are not "
                  f"corrected: with {n_tests} tests, about {0.05 * n_tests:.0f} will fall below .05 by chance, so "
                  "trust an effect that repeats across properties or matches a prediction more than a single hit.",
                  "", "| Property | Factor | Noticed | p | Chose high (of those who noticed) | p |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for row in differs.itertuples():
            groups = sub[(sub["property"] == row.property) & (sub["factor"] == row.factor)]
            seen = ", ".join(f"{g.group} {num(g.noticed_share, '.0%')} (n={g.n})" for g in groups.itertuples())
            high = ", ".join(f"{g.group} {num(g.prefer_high_share, '.0%')} (n={g.pref_n})" for g in groups.itertuples())
            lines.append(f"| {summary.loc[row.property, 'name']} | {row.factor} | {seen} | "
                         f"{num(row.p_noticed_differs, '.3f')} | {high} | {num(row.p_prefer_high_differs, '.3f')} |")
    lines.append("")
    lines += [f"Rule (from the config): continuous or ordinal; noticed by at least {config['noticed_min_share']:.0%} "
              f"after subtracting the false-alarm rate; then the top {config['select_n']} by importance (BWS score "
              "on 0..1) x heterogeneity H. P(selected) is the share of "
              f"{config.get('selection_bootstrap', 0)} bootstrap samples of participants in which the property is "
              "selected.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", help="data files (default: files_heroku in path_data)")
    parser.add_argument("--out", default=config["output_dir"], help="output folder")
    parser.add_argument("--public", help="folder with the published CSV files (export_public.py) instead of raw data")
    args = parser.parse_args()

    if args.public:
        # the published, anonymised data: the participants the original analysis kept
        from avprops.publish import load_public
        records = load_public(resolve(args.public), [q["id"] for q in load_questionnaires(config)])
        result = analyse(records, load_properties(config), config, load_questionnaires(config), load_scenes(config))
        write_outputs(result, config, resolve(args.out))
        return
    if args.data:
        files = [resolve(f) for f in args.data]
    else:
        files = [resolve(config["path_data"]) / f for f in config["files_heroku"]]
        if config["download_data"]:
            download(load_secret()["mongodb_uri"], files[0])
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"data file(s) not found: {', '.join(missing)}")

    records = rows_to_records(read_rows(files), config["attention_max_wrong"])
    if not records:
        raise SystemExit("no participant data in " + ", ".join(str(f) for f in files))
    result = analyse(records, load_properties(config), config, load_questionnaires(config), load_scenes(config))
    write_outputs(result, config, resolve(args.out))


def write_outputs(result: dict, config: dict, out) -> None:
    out.mkdir(parents=True, exist_ok=True)
    result["participants"].to_csv(out / "participants.csv", index=False)
    result["summary"].to_csv(out / "summary.csv", index_label="property")
    result["subgroups"].to_csv(out / "subgroups.csv", index=False)
    result["open_answers"].to_csv(out / "open_answers.csv", index=False)
    write_report(result, config, out / "report.md")

    print(f"{result['n_kept']} of {result['n_records']} participants kept; "
          f"false alarms {result['false_alarm_rate']:.0%}, "
          f"prefer no display {result['prefers_no_display']:.0%}")
    print(result["summary"][["name", "bws_score", "noticed_corrected", "H", "score", "p_selected", "status"]]
          .to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
