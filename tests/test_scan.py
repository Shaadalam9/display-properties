import numpy as np

from avprops.scan import ScanSettings, frame_risk, pick, windows, WIDTH


def samples(risks, cuts=(), n=4):
    return [{"t": k / 2.0, "risk": r, "ahead": 1, "vru": 0, "n": n,
             "hist_corr": 0.1 if k in cuts else 0.99} for k, r in enumerate(risks)]


def test_frame_risk_counts_only_road_users_ahead_and_weights_vulnerable_ones():
    s = ScanSettings()
    h = 360
    centre = WIDTH / 2
    car = [centre - 40, 200, centre + 40, 300]           # ahead, 100 px tall
    walker = [centre - 10, 220, centre + 10, 300]        # ahead, 80 px tall, vulnerable
    aside = [10, 100, 60, 340]                            # far left: outside the corridor
    risk, ahead, vru = frame_risk(np.array([car, walker, aside]), np.array([2, 0, 0]), h, s)
    assert ahead == 2 and vru == 1
    assert np.isclose(risk, max(100 / h, s.vru_weight * 80 / h))


def test_windows_skip_cuts_and_edges():
    s = ScanSettings(skip_s=10, window_s=14, step_s=2)
    risks = [0.05] * 240                                  # 120 s at 2 fps
    risks[100:110] = [0.6] * 10                           # risky stretch at 50-55 s
    recs = samples(risks, cuts={180})                     # a cut at 90 s
    scored = windows(recs, 120.0, s)
    assert all(10 <= w["start_s"] and w["end_s"] <= 110 for w in scored)
    assert any(w["cut"] for w in scored if w["start_s"] <= 90 < w["end_s"])
    high = pick(scored, 1, low=False)[0]
    assert high["start_s"] <= 50 and high["end_s"] >= 55
    low = pick(scored, 2, low=True, min_gap_s=30)
    assert len(low) == 2 and abs(low[0]["start_s"] - low[1]["start_s"]) >= 30
    assert not any(w["cut"] for w in low + [high])


def test_calm_windows_need_some_traffic():
    s = ScanSettings(skip_s=0, window_s=14, step_s=2)
    empty = samples([0.0] * 120, n=0)                     # nobody on the road at all
    assert pick(windows(empty, 60.0, s), 3, low=True) == []
