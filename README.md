# Which visual properties matter in AV explanation displays

Crowdsourced study to decide, from the literature and from participants watching driving clips, which visual
properties of an automated-vehicle (AV) explanation display are worth personalising. The result is a ranked set of
about four properties, their usable ranges, and fixed values for everything else. A later preference optimisation
then tunes only the selected properties.

The code implements the study design: a literature review gives the candidate properties, a crowdsourced study
measures which of them matter and which people disagree about, and pre-registered rules select the final set. The
crowdsourcing setup follows
[bazilinskyy/ehmi-vehicles](https://github.com/bazilinskyy/ehmi-vehicles): a [jsPsych](https://www.jspsych.org) 8
experiment served by Node.js (Express) and stored in MongoDB, deployed on Heroku, with the analysis in Python.

## Experiment
The experiment is in `public/views/experiment.html` and `public/js/experiment.js`. `app.js` serves it and stores the
data in the `entries` collection of a MongoDB database. jsPsych and its plugins are installed from npm and served by
`app.js`.

### Procedure
1. The clips of the participant are preloaded and the browser switches to full screen.
2. Device check: phones, tablets and screens smaller than 1280 × 720 px cannot continue.
3. Welcome, information sheet and consent. Declining ends the study.
4. Demographics (age, nationality and country of residence from an ISO 3166 list, gender, driving licence, car use,
   experience with driver assistance, self-driving rides, familiarity with head-up displays, colour vision
   deficiency).
5. Questionnaires (`study/questionnaires.json`): propensity to trust automation (Körber 2019, 3 items) and affinity
   for technology interaction (ATI, Franke et al. 2019, 9 items), to test whether preferences split by user type.
6. **Comprehension**: one short clip per scene that has a question (not the baseline scene), showing one property at a
   random level. The clip stops before the car acts and the participant answers "What will the car do next?". This
   comes before any property is named, so nothing but the display tells them the answer.
7. **Task C**: a baseline video of the display, then the open question "What would you want this display to show, or
   show differently?", asked before any property is named.
8. **Task A**: Best-Worst Scaling (jsPsych MaxDiff). Each trial shows 4 properties; the participant picks the most and
   least important for understanding what the car is doing and why. Each property comes with a short looping preview
   of its levels side by side (`public/stimuli/previews/`), so it is judged from what it looks like and not only from
   its name. With the 17 screened-in properties that is 13 trials.
9. **Task B**: A/B video pairs. For each of 10 properties, the same situation with the property at its low and high
   level (side randomised): did you notice a difference, which would you prefer while riding, how much does it matter
   (1–7), and for each video understanding, perceived safety and mental load (1–7). Mixed in: one pair of the display
   vs the same clip with no display. At the end, 3 of the pairs again with the sides swapped, to measure how
   consistent each participant is.
10. Five attention checks are mixed in between the trials of Tasks A and B; three wrong answers end the study
    (`finish_fail`).
11. An optional final comment, then the finish page with the participant's completion code.

Data is saved after every trial, so the answers of participants who stop early are kept. The order of trials, the BWS
version, the scene per property and the side of each level are seeded with the worker code.

### Stimuli
`study/properties.json` lists the candidate properties from the literature review
(`reports/AV display property literature.md`): the 17 screened in, 3 reserves, and the ones fixed or excluded with the
reason. The screened-in set includes the four parameters of the follow-up preference study (highlight strength
`mask_alpha`, path opacity `trajectory_alpha`, background dimming `background_dim_alpha` and colour palette
`palette_id`), with the same ranges and defaults, so that study's choice of parameters can be checked against what
participants rank here. `study/scenes.json` lists the driving situations. Every scene needs, under `public/stimuli/<scene id>/`:
`baseline.mp4`, `none.mp4` (the clip without a display), and `<property id>_low.mp4` plus `<property id>_high.mp4` for
every screened-in property. A property with more than two levels lists `variants` and has one
`<property id>_high_<variant>.mp4` per variant (the palette: the standard palette against the Okabe-Ito, minimalist
and high-saliency palettes); each participant sees one variant, at random. The two clips of a pair must differ in that
one property only.

Each scene records the `country` it was filmed in (ISO code) and its `criticality` (low or high risk), so the
analysis can compare preferences between risky and calm situations, and between clips from a participant's own
country and from elsewhere. For both comparisons to work, include scenes of both criticality levels and from more
than one country. Properties whose two clips differ mainly in hue are marked `colour_dependent`.

The clips are made, not filmed: a real dashcam clip with the explanation display drawn on top by
`avprops/clip_render.py`. Every property is a parameter of that renderer, and each property in `properties.json` has a
`render` block with the settings of its low and high clip; everything else stays at the `baseline`. A property without a
`render` block cannot be made as a clip and must stay screened out, and `tests/test_stimuli.py` checks that the low
and high clip of every renderable property actually differ on screen.

The renderer draws from per-frame overlay geometry: the planned path, the road users with outlines, boxes, distances
and track ids, an occlusion map and the explanation ramp (format documented in `avprops/clip_render.py`). The scenes
use geometry from a full perception pipeline (`data/geometry_pipeline/`): YOLO26x segmentation with ByteTrack,
SegFormer road segmentation, Depth Anything V2 distances, a camera auto-calibration per clip, and a planned path
anchored to the road from visual odometry of the car's actual motion. Each scene's `geometry` names its file; a
`camera` in the scene is only given where the auto-calibration found none, and then wins over the geometry's.
Geometry made from footage above 1280 px wide is scaled down on reading, so every stimulus is 1280×720.
`extract_geometry.py` is a simpler fallback with YOLO segmentation and ByteTrack and a straight path in the car's
lane.

```bash
uv sync --extra extract                                  # detection and tracking dependencies
uv run python extract_geometry.py --device mps           # data/geometry/<scene>_geometry.jsonl.gz (cuda:0 on a GPU host)
uv run python render_stimuli.py --calibrate              # check each scene's camera constants
uv run python render_stimuli.py --preview                # a still of every clip, as a contact sheet
uv run python render_stimuli.py --check                  # how much each pair differs in each scene
uv run python make_design.py                             # uses the check to pick scenes per property
uv run python render_stimuli.py                          # the clips, as H.264, into public/stimuli/
uv run python render_stimuli.py --previews               # the BWS previews, from the rendered clips
uv run python make_design.py                             # again, so the design lists the previews
```

A property can only differ in a scene that has what it acts on: other road users' paths need people moving sideways,
colour by type needs vehicles, "when shown" needs an event with a few seconds before it. `--check` measures, per scene,
the share of pixels that differ between the low and high clip over the whole clip and flags pairs below 0.2% as
`WEAK`. It saves the result to `data/stimulus_check.json`; `make_design.py` then lets each property's A/B pair use
only the scenes where it passed, and stops with an error if a property passes in none. The threshold is a screen for
pairs with nothing to show, not proof that people notice the difference; the "did you notice" answers in the pilot do
that.

Because the pipeline's path covers only where the car actually drives, it is short or absent while the car stands
still, so the path properties pass the check mostly in scenes where the car moves. The highlight properties act on
pedestrians (the road users the car reacts to) and so pass mostly in the risky scenes. Both limit how far the calm vs
risky comparison can be read for those properties; `make_design.py` prints the scenes each property may use. Other
road users' paths show only sideways movement, because the camera car's own speed is not subtracted.

Source clips go in `data/clips/` and, like the rendered stimuli, are not in git.

**Finding scenes.** The scenes come from long dashcam drives in many cities (the CROWD dataset), one calm and one
risky stretch per city. `scan_videos.py` proposes them:

```bash
uv run --extra extract python scan_videos.py VIDEOS... --segments segments.csv --device cuda:0
```

It decodes each video at 2 frames per second, detects road users with YOLO and scores every 14 s window by how
close and central the road users ahead are (pedestrians, cyclists and motorcyclists weigh 1.5 times as much),
leaving out windows with a cut, dark windows (dusk, night) and, with `--segments`, anything outside the stretches the
dataset's mapping lists as car driving in the locality. It writes the calmest (with some traffic) and riskiest
windows of each video with a strip of frames to review. The score is a screen, not a measure of danger: big trucks
close ahead score high even in a queue, and a video's label can be wrong (one "car" video turned out to be a
walking tour), so every scene is chosen by eye. Scenes record their `source` video and start, `country` and
`criticality`, and `texts_verified` stays false until someone has checked the explanation texts against the clip. `make_design.py` lists the clips the
design needs that are still missing.

After any change to the properties, scenes or settings, regenerate the design:

```bash
uv run python make_design.py
```

This writes `public/design.json`: the balanced BWS design (50 versions), the properties and scenes, and the A/B and
attention settings.

### Running the experiment
Install the dependencies with `npm install` and start the server with
`CONNECTION='<MongoDB connection string>' npm start`, then open http://localhost:3000. Without `CONNECTION` the server
stores the data in `data/entries.jsonl`, in the same shape as a MongoDB export, so the study can be tried without a
database. URL parameters:
* `debug=true`: log details to the browser console, skip the device check, allow continuing before a video ends, show
  the names of missing clips, and mark the data as debug (excluded from the analysis).
* `save_data=false`: do not send data to the server.
* `worker_code=<code>`: participant code to store with the data and show on the finish page. `PROLIFIC_PID` is used
  when present; otherwise a code is generated.

**Sample.** To compare regions, recruit separate groups per region with the platform's screening (on Prolific:
the "Current country of residence" prescreener, one study per region with its own quota) and keep the groups
roughly equal, rather than relying on whoever signs up. Prolific's pool is mostly Europe and North America;
regions where it has few participants, such as Egypt, may need another crowdsourcing platform, using the same
`worker_code` link.

The live study runs on Heroku (`Procfile`, Node.js 22), with the connection string set as the `CONNECTION` config
variable.

## Analysis
The analysis is in the `avprops` package, started by `analysis.py`. Tested with Python 3.12.13 and
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python analysis.py
```

### Configuration
Settings are read from `config` and secrets from `secret` in the repository folder. Both are not in git: create them
from `default.config` and `default.secret`. If there is no `config`, `default.config` is used.

`config`:
* `path_data`, `files_heroku`: folder and files with the data exported from MongoDB.
* `download_data`: export the data from MongoDB into the first of `files_heroku` before the analysis (needs
  `mongoexport` and `mongodb_uri` in `secret`).
* `bws_items_per_trial`, `bws_appearances_per_item`, `bws_design_versions`: the BWS design.
* `ab_pairs_per_participant`: A/B pairs per participant (10; 0 = all properties); `ab_repeats`: pairs repeated with
  the sides swapped (3); `ab_no_display`: add the display vs no display pair; `comprehension`: ask "what will the car
  do next?" in the scenes that have a `comprehension` block; `questionnaires`: which questionnaires to ask;
  `ab_collect_ratings`: ask the three
  ratings for both videos.
* `attention_fail_at`: wrong attention answers that end the study.
* `colour_properties_without_cvd`: leave participants who report a colour vision deficiency out of the main results
  of `colour_dependent` properties (they are still compared in `subgroups.csv`).
* Exclusions: `exclude_debug`, `exclude_unfinished`, `attention_max_wrong`, `min_median_bws_rt_ms`,
  `min_repeat_consistency` (0 = report consistency only),
  `max_hidden_ab_share` (share of A/B pairs during which the participant left the page).
* Selection rules: `select_n` (4), `noticed_min_share` (0.70), `matters_top_fraction` (0.5),
  `disagreement_max_majority_share` (0.75), `redundancy_min_abs_r` (0.70). Set them in the pilot, then pre-register.

`secret`:
* `mongodb_uri`: connection string of the MongoDB database, including the password.

### Running the analysis
`uv run python analysis.py` reads `files_heroku` in `path_data`; `--data <file>` reads other files, for example
`data/entries.jsonl` from a local run. To export the data by hand:
`mongoexport --uri <mongodb_uri> --collection entries --type json --out data/heroku_0.json`.

Outputs go to `_output/` (not in git, as they hold personal data):
* `participants.csv`: every participant: finished, attention answers, questionnaire scores, repeat consistency,
  whether they preferred no display, comprehension, excluded and why.
* `summary.csv`: per property the BWS score with bootstrap CI, share noticing, preference split, importance, rating
  differences with Wilcoxon tests, comprehension after the low vs the high clip (Fisher's exact test), the
  correlation of choosing the high level with each questionnaire score, and its status. Repeated pairs and the
  no-display pair are left out of these.
* `subgroups.csv`: per property, noticing and preference in low vs high criticality scenes, in clips from the
  participant's own country vs elsewhere, and (colour-dependent properties) with vs without a colour vision
  deficiency, each with Fisher's exact test.
* `redundant.csv`, `open_answers.csv` (for thematic coding of Task C) and `report.md`, which also gives the overall
  repeat consistency, the share of participants who preferred no display, the countries of residence and the
  subgroup differences.

Every property gets a status saying where it stopped: `not noticed`, `does not matter`, `consensus: fix at low/high`,
`redundant with …`, `eligible, below cut-off`, or `selected`.

To check the analysis before real data exists, `simulate.py` writes synthetic participants with a known truth in the
same format as the server:

```bash
uv run python simulate.py --n 150
uv run python analysis.py --data data/simulated/entries.jsonl --out _output/simulated
```

### Tests

```bash
uv run pytest
```
