"""Replace the IP addresses in the raw data file by placeholders, once they are no longer needed.

    uv run python scrub_ips.py --data /srv/study-data/entries.jsonl            # writes entries.scrubbed.jsonl
    uv run python scrub_ips.py --data /srv/study-data/entries.jsonl --in-place # replaces the file

The server stores each participant's IP address only to find people who take part more than once
(data management plan, DMP-3509). Run this when data collection and payment are finished and before
the raw data are archived: every address becomes a placeholder (0.0.0.1, 0.0.0.2, ... in order of
first appearance), the same one for the same address, so the duplicate check still gives the same
result on the scrubbed file, but the addresses themselves are gone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from avprops.common import resolve
from avprops.crowd import read_documents


def scrub(documents: list[dict]) -> tuple[list[dict], int]:
    """The documents with every ``ip`` replaced by a placeholder, and the number of addresses."""
    masked: dict[str, str] = {}
    out = []
    for doc in documents:
        ip = doc.get("ip")
        if ip and not str(ip).startswith("0.0.0."):
            doc = dict(doc, ip=masked.setdefault(ip, f"0.0.0.{len(masked) + 1}"))
        out.append(doc)
    return out, len(masked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="the data file the server writes (JSON lines)")
    parser.add_argument("--in-place", action="store_true", help="replace the file instead of writing a copy")
    args = parser.parse_args()

    path = resolve(args.data)
    documents, n = scrub(read_documents(path))
    target = path if args.in_place else path.with_name(path.stem + ".scrubbed" + path.suffix)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(doc) + "\n" for doc in documents), encoding="utf-8")
    tmp.replace(target)
    print(f"{n} IP addresses replaced by placeholders in {len(documents)} saved batches -> {target}")


if __name__ == "__main__":
    main()
