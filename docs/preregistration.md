# Pre-registration (draft)

**Which visual properties of an AV explanation display are worth personalising? A crowdsourced study**

Draft written from the code and settings that will run the study, so every number below is the one the analysis
uses. Items marked **[decide]** still need a decision; after the pilot, the final version is frozen on OSF together
with the commit hash of this repository and the settings fingerprint that `analysis.py` prints in `report.md`
(currently `default.config (sha256 2214130e3a01)`, it changes with any setting).

## 1. Research questions

* **RQ1.** Which visual properties of an AV explanation display do people consider most important for understanding
  what the vehicle does and why? (Task A, Best-Worst Scaling)
* **RQ2.** For which properties do people notice the difference between two levels, care about it, and disagree about
  the preferred level, so that personalising it is worthwhile? (Task B, A/B video pairs)
* **RQ3.** Do noticing and preferences differ with the criticality of the driving situation, between clips from the
  participant's own country and elsewhere, with colour vision, and with propensity to trust automation and affinity
  for technology interaction?
* **RQ4.** Do the four parameters of the follow-up preference study (highlight strength, path visibility, background
  dimming, colour palette) rank among the properties participants prioritise?

The main outcome is the set of (at most) four properties selected by the rule in section 6. The study is exploratory
in the sense that it does not test directional hypotheses; what is fixed in advance is the selection rule, the
exclusions and the sample size.

## 2. Design

Within-participants online experiment (jsPsych), about 30 minutes, on a computer (screens of at least 1280 × 720 px;
phones and tablets cannot take part).

**Stimuli.** 14-second clips cut from public dashcam footage (CROWD dataset), with an explanation display rendered
on top, in one calm and one risky scene per city (18 scenes, 9 countries; Mexico City has only a calm scene). For each property and scene there is a
low and a high clip, identical except for that one property. Participants are told the footage was filmed by people
driving and asked to imagine the car drives itself.

**Properties.** 17 properties, each at two levels (`study/properties.json`):

* tunable (eligible for selection), continuous: path visibility, path width, path length, path certainty colour,
  highlight strength, pedestrian marker size, background dimming; ordinal: explanation detail, confidence indicator;
* categorical (settled by majority, never selected): path style, manoeuvre cue, which road users are highlighted,
  highlight style, colour by road-user type, other road users' paths, when the display is shown, colour palette
  (three alternatives to the standard palette; each participant sees one at random).

**Assignment.** 500 participant plans are generated in advance (`design.json`, seed 2026) and handed out in turn by
the server (participant 1 gets plan 0, the next plan 1, and so on), so every run of consecutive participants is
counterbalanced: each block of 50 plans uses each of the 50 BWS versions once, and the properties a participant
compares, the scenes each is compared in, the palette alternative, the catch pairs and the comprehension clips' property
and level are each spread evenly over the plans (least used so far, ties at random). A plan fixes the BWS version, the
order and content of all A/B pairs and the comprehension clips. The logged plan is checked against what the participant
actually saw (`plan_mismatches`). The task order is fixed (comprehension, baseline, BWS, then A/B pairs): the BWS
ranking comes before the A/B pairs so that it is not swayed by the few properties a participant happens to compare.

**Procedure.**

1. Information, consent (18+), device check, demographics (age, gender, nationality, country of residence, driving
   experience; colour vision only after a separate, explicit consent for this health-data item, with "prefer not to
   say"; without it colour vision counts as unknown).
2. Questionnaires: propensity to trust automation (Körber, 2019) and affinity for technology interaction (ATI;
   Franke et al., 2019).
3. Comprehension: short clips, in random order, that stop before the event; "what will the car do next?" (properties marked
   `in_comprehension: false`, explanation detail and manoeuvre cue, are left out because they name the answer).
4. A baseline clip with the default display and an open question: what would you want this display to show, or show
   differently? (required; analysed qualitatively, not part of the rule)
5. **Task A, BWS:** 13 trials of 4 properties (each property shown 3 times, 50 balanced versions), each with a
   preview of its two levels; participants pick the most and the least important for understanding what the car does
   and why.
6. **Task B, A/B pairs:** 17 pairs, each two clips side by side, played for at least 14 s before answering:
   6 tunable properties × 2 different scenes, 2 categorical properties × 1 scene, 2 catch pairs (identical clips), and
   1 pair of display vs no display. Per pair: "Did you notice a difference?" (yes/no), preference on a 5-point scale
   (A clearly, A slightly, no preference, B slightly, B clearly), and how much the difference matters (7 points). Sides
   are counterbalanced: a tunable property has its high level on A in one scene and on B in the other, and half of a
   participant's pairs have the high level (or the display) on A. The pairs are in random order, with two pairs of the
   same property, or of the same scene, never one after the other, and the catch pairs not among the first two and not
   next to each other.
7. Six attention checks, spread out: the BWS trials and A/B pairs are cut into six equal parts and one check goes
   before a random trial of each part; the study ends after 3 wrong
   answers (partial payment).
8. Optional final comment.

## 3. Sample size and stopping rule

Target: **600 usable participants** after exclusions, expected to need about **800 recruited** (Microworkers,
worldwide).

Justification: simulations of the full pipeline (`simulate.py`), with a population mixing consensus properties,
properties with stable opposite preference types, and indifferent participants (15%), 10% false alarms and scene
noise, give these error rates of the selection rule:

| Usable N | Any wrong pick | Clearly wrong pick | Missed a true property | Exactly the right four |
|---|---|---|---|---|
| 300 | 45% | 20% | 10% | 51% |
| 450 | 26% | 6% | 12% | 62% |
| 600 | 18% | 5% | 8% | 78% |
| 800 | 20% | 5% | 10% | 70% |
| 1000 | 18% | 4% | 14% | 71% |

Errors drop steeply up to about 600 and hardly after. **Stopping rule:** recruit in batches; after each batch count the
usable participants with the exclusions of section 4 (without looking at the results); stop when 600 are reached, or
at **[decide: a cap, e.g. 1000 recruited or the budget]**, whichever comes first. The pilot (about 20 participants,
own `study_id`) is not part of the sample.

## 4. Exclusions

A participant is excluded if any of these holds (`apply_exclusions`, settings in `default.config`):

* debug session, or another `study_id` or design version;
* did not finish (left before the last page, or stopped by the attention checks);
* a finished record that lacks the consent, any BWS trial or any A/B pair;
* a later session from an IP address whose earlier session ended (finished or stopped by the attention checks); the
  first such session is kept;
* more than 1 wrong attention answer;
* median BWS response time below 1500 ms;
* the browser tab hidden during more than half of the A/B pairs;
* rows from more than one browser session, or finished in less than 8 minutes by the server's clock (server session
  token);
* noticing a difference in catch pairs: **not** an exclusion (`max_catch_noticed` off); instead the false-alarm rate is
  subtracted from noticing (section 6).

Participants who report a colour vision deficiency are kept, but left out of the main results of the colour-dependent
properties (path certainty colour, colour by road-user type), and compared separately (RQ3). For the colour palette
they are kept in, as people with a colour vision deficiency are part of that question.

## 5. Measures

* **BWS score** per property: per participant (times best − times worst) / times shown, averaged over participants,
  with a bootstrap 95% CI (2000 samples).
* **Noticing** per property: share of its pairs in which a difference was reported, minus the false-alarm rate (share
  of catch pairs in which a difference was reported, pooled over participants).
* **Preference**: share preferring the high level, the low level, no preference; mean graded preference.
* **Heterogeneity H** per property: one-way intraclass correlation of the direction of each participant's two choices
  (high = 1, low = −1), among participants who chose a level in both scenes (no-preference answers left out). H is
  near 0 when people agree or choose at random, and near 1 when people hold stable but opposite preferences.
* **Importance** (7 points) per pair; **comprehension** correct/incorrect; questionnaire scores.

## 6. Primary analysis: the selection rule

1. Only tunable properties (continuous or ordinal) are eligible.
2. A property qualifies if its corrected noticing is at least **0.70** and its **H is at least 0.30**.
3. Qualifying properties are ranked by **score = importance × H**, with importance = (BWS score + 1) / 2 on 0..1.
4. The **top 4** are selected (fewer if fewer qualify).
5. A bootstrap over participants (500 samples) gives each property's probability of being selected; it is reported
   but does not change the selection.

Every other property gets a status: *not noticed*; *people agree: fix at* the majority level (tunable, noticed, H below
0.30); *categorical: settle at* the majority level; the colour palette is settled per alternative (each palette against
the standard one, not pooled); *eligible, below cut-off*.

## 7. Secondary analyses

* **Display vs no display:** share preferring no display at all.
* **Comprehension:** per property, share correct after the low vs the high clip (between participants, Fisher's exact
  test).
* **User types (RQ3):** per property, point-biserial correlation between choosing the high level and the trust and ATI
  scores.
* **Subgroups (RQ3):** per property, noticing and preference in calm vs risky scenes, own country vs elsewhere, and
  with vs without a colour vision deficiency (colour-dependent properties), Fisher's exact test.
* **RQ4:** the BWS ranks and selection statuses of highlight strength, path visibility, background dimming and colour
  palette.

Secondary tests are reported with their p-values, without a correction, and described as exploratory.

## 8. Data, code and deviations

The experiment, simulation and analysis code are in this repository; the anonymised data (new participant IDs, age
in bands, redacted open answers, exclusion flags, no IP addresses) are published on 4TU.ResearchData, and
`analysis.py --public` reproduces the analysis from them. Any deviation from this plan is reported as such in the
paper, with its reason.

## Before freezing [decide]

* the recruitment cap in section 3;
* whether the pilot leads to changes (they would be listed here, and the settings fingerprint updated);
* where to register (OSF or AsPredicted) and the planned registration date, before the first main-study participant.
