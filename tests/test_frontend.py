"""The front end and the Python side agree (review by Mark Colley, "equivalence plan").

* A golden trace (tests/fixtures/golden_trace.jsonl, recorded with tests/frontend.py from the
  real experiment in a headless browser) is parsed into a complete record that matches the plan
  it logged, has every field simulate.py writes, and goes through the analysis.
* The live tests run the current front end again (so a change to it is checked at once), and
  once more with failing and delayed saves: every trial must still be stored.
"""

import json
import time

import pytest

from avprops.analysis import analyse
from avprops.common import load_config, load_properties
from avprops.crowd import plan_mismatches, read_rows, rows_to_records
from tests.frontend import TRACE, available, run_session, server

CONFIG = load_config()


def record_from(path):
    records = rows_to_records(read_rows([path]), CONFIG["attention_max_wrong"])
    assert len(records) == 1
    return records[0]


def assert_complete(record):
    """A whole session, as the plan it logged says, with every field the simulator writes."""
    import simulate
    design = json.loads((TRACE.parents[2] / "public" / "design.json").read_text())
    assert record["finished"] and record["consent"]
    assert record["attention"]["shown"] >= 5 and record["attention"]["passed"]
    assert plan_mismatches(record) == []
    assert len(record["bws"]) == len(design["bws"]["versions"][0])
    simulated = rows_to_records([row for doc in simulate.simulate(design, 1, 0, CONFIG)[0] for row in doc["data"]],
                                CONFIG["attention_max_wrong"])[0]
    assert set(simulated) <= set(record), f"missing in the real record: {set(simulated) - set(record)}"
    for part in ("bws", "ab", "comprehension"):
        missing = set(simulated[part][0]) - set(record[part][0])
        assert not missing, f"{part}: fields the simulator writes but the front end does not: {missing}"
    for key in ("age", "nationality", "country_residence", "gender", "colour_vision_deficiency"):
        assert key in record["demographics"]


def test_golden_trace_is_a_complete_session_that_follows_its_plan():
    record = record_from(TRACE)
    assert_complete(record)
    result = analyse([dict(record, debug=False)], load_properties(CONFIG), CONFIG)
    assert result["n_records"] == 1 and len(result["summary"]) == len(load_properties(CONFIG))


browser = pytest.mark.skipif(bool(available()), reason=available() or "browser available")


@browser
def test_live_session_matches_its_plan(tmp_path):
    data = tmp_path / "entries.jsonl"
    with server(data) as url:
        screens = run_session(url, "livecheck")
    assert screens.count("bws") == 13 and screens.count("pair") == 17
    assert_complete(record_from(data))


@browser
def test_saves_survive_network_faults(tmp_path):
    calls = {"n": 0}

    def flaky(route, request):
        calls["n"] += 1
        if calls["n"] <= 3:                       # the first save fails on all three attempts
            route.fulfill(status=500, body="down")
            return
        if '"task":"final"' in (request.post_data or ""):
            time.sleep(3)                         # the last save is slow, the page must wait for it
        route.continue_()

    data = tmp_path / "entries.jsonl"
    with server(data) as url:
        run_session(url, "faultcheck", route=flaky)
    record = record_from(data)
    indices = sorted({row["trial_index"] for row in read_rows([data])})
    assert indices == list(range(indices[-1] + 1)), "trials are missing"
    assert record["finished"] and record["consent"]
