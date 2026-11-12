"""Run the real experiment in a headless browser, for the front-end tests (review by Mark Colley).

The study runs as participants get it: app.js serves it, a headless Chromium (Playwright) opens it
in debug mode (which skips the device check and the waiting for videos, nothing else) and an
automatic responder answers every screen. What the server stores is then read by the same code as
the real data. Without Playwright, its browser or Node.js the tests that need them are skipped.

    uv run python tests/frontend.py        # record a new golden trace into tests/fixtures/
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "tests" / "fixtures" / "golden_trace.jsonl"

# One step of the automatic responder: answer whatever screen is showing and return what it was.
# Attention checks get the right answer; everything else gets a valid one.
STEP = r"""() => {
  const q = s => document.querySelector(s);
  const all = s => [...document.querySelectorAll(s)];
  const content = q('#jspsych-content');
  if (!content) return 'waiting';
  const click = s => { const el = q(s); if (el && !el.disabled) { el.click(); return true; } return false; };
  if (click('#jspsych-fullscreen-btn')) return 'fullscreen';
  if (q('[id^="jspsych-maxdiff"]')) {
    const r = all('#jspsych-content input[type=radio]'); r[0].click(); r[r.length - 1].click();
    click('#jspsych-maxdiff-next'); return 'bws';
  }
  if (q('.pair')) {
    const names = [...new Set(all('#jspsych-content input[type=radio]').map(i => i.name))];
    names.forEach((n, k) => {
      const opts = all(`#jspsych-content input[name="${n}"]`);
      opts[(k * 2 + 1) % opts.length].click();
    });
    return click('#jspsych-survey-html-form-next') ? 'pair' : 'pair-waiting';
  }
  if (q('select[name="nationality"]')) {
    q('select[name="nationality"]').value = 'NL'; q('select[name="country_residence"]').value = 'NL';
    click('#jspsych-survey-html-form-next'); return 'countries';
  }
  if (q('.jspsych-survey-likert-statement')) {
    all('.jspsych-survey-likert-opts').forEach(row => row.querySelector('input').click());
    click('#jspsych-survey-likert-next'); return 'questionnaire';
  }
  if (q('.jspsych-survey-multi-choice-question')) {
    const right = (window.ATTENTION_QUESTIONS || []).map(a => a.correct);
    all('.jspsych-survey-multi-choice-question').forEach(question => {
      const options = [...question.querySelectorAll('input')];
      (options.find(o => right.includes(o.value)) || options.find(o => o.value === 'Yes') || options[0]).click();
    });
    click('#jspsych-survey-multi-choice-next'); return 'choice';
  }
  if (q('#jspsych-survey-text-next')) {
    all('#jspsych-content input[type=text], #jspsych-content input[type=number], #jspsych-content textarea')
      .forEach(el => { el.value = el.dataset.name === 'age' || el.type === 'number' ? '30' : 'Show why it brakes.'; });
    click('#jspsych-survey-text-next'); return 'text';
  }
  if (click('#jspsych-instructions-next')) return 'instructions';
  const button = q('#jspsych-content .jspsych-btn');
  if (button && !button.disabled) { button.click(); return 'video'; }
  return 'waiting';
}"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def available() -> str:
    """Why the browser tests cannot run here ("" when they can)."""
    if shutil.which("node") is None:
        return "Node.js is not installed"
    if not (ROOT / "node_modules" / "jspsych").exists():
        return "npm install has not been run"
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            p.chromium.launch().close()
    except Exception as err:  # noqa: BLE001 - any failure means: skip
        return f"Playwright or its browser is missing ({type(err).__name__})"
    return ""


@contextmanager
def server(data_file: Path) -> Iterator[str]:
    """app.js on a free port, storing into data_file; yields its address."""
    port = free_port()
    env = dict(os.environ, PORT=str(port), DATA_FILE=str(data_file))
    env.pop("CONNECTION", None)
    proc = subprocess.Popen(["node", "app.js"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(url + "/", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("the study server did not start: " + proc.stderr.read().decode()[-500:])
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def run_session(url: str, worker_code: str, route: Callable | None = None, timeout_s: float = 240) -> list[str]:
    """Take part once, from the welcome page to the finish page; returns the screens answered.
    ``route`` (Playwright route handler) can intercept the saves, e.g. to make them fail."""
    from playwright.sync_api import sync_playwright
    screens = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("dialog", lambda dialog: dialog.accept())  # the "problem while saving" alert
        if route:
            page.route("**/experiment-data", route)
        page.goto(f"{url}/?debug=true&platform=test&worker_code={worker_code}")
        deadline = time.time() + timeout_s
        while "/finish" not in page.url:
            if time.time() > deadline:
                raise TimeoutError(f"the session did not finish; last screens: {screens[-5:]}")
            try:
                screens.append(page.evaluate(STEP))
            except Exception:  # noqa: BLE001 - the page is navigating to the finish page
                pass
            page.wait_for_timeout(150)
        page.wait_for_timeout(500)
        browser.close()
    return [s for s in screens if s != "waiting"]


def record_trace() -> None:
    """Record a new golden trace: one session's rows, as the server stores them."""
    data = TRACE.with_suffix(".tmp")
    data.unlink(missing_ok=True)
    with server(data) as url:
        run_session(url, "goldentrace")
    lines = [line for line in data.read_text().splitlines() if line.strip()]
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    TRACE.write_text("\n".join(lines) + "\n")
    data.unlink()
    print(f"{len(lines)} saved batches -> {TRACE}")


if __name__ == "__main__":
    record_trace()
