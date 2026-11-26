# Microworkers campaign

Text and settings for the campaign, pilot first. Fill in the **[brackets]**; the amounts come out of the pilot.

## Settings

| Setting | Pilot | Main study |
|---|---|---|
| Campaign type | Basic campaign | Basic campaign |
| Positions | 20 | in batches of about 200, until 600 usable participants (see `docs/preregistration.md`, section 3) |
| Payment per task | **[amount]** | **[amount]** |
| Time to finish (worker) | 60 minutes | 60 minutes |
| Countries | all, except China, Russia, Iran, North Korea | same |
| Study link | `https://[app].herokuapp.com/?platform=microworkers` | same |
| `study_id` in `default.config` | `av_display_properties_pilot` (then `make_design.py`) | `av_display_properties_v1` |
| Proof | text field: the completion code | same |

The link carries no worker ID: the study makes a random completion code for each participant and shows it at the
end. Only Microworkers can link a code to a worker.

**Partial payment** (participants stopped by the attention checks also get a code, on the "too many incorrect
answers" page): check before the pilot how Microworkers allows paying them less than the full amount, e.g. a lower
task payment plus a bonus for full completions, or rating them satisfied at the full amount if the difference is
small. **[decide]**

## Campaign title

Watch short traffic videos and tell us which display features matter to you (about 30 minutes, computer only)

## What is expected from the worker

> This is a study by researchers at University College London (UK) about displays in future
> self-driving cars. You will watch short videos of real traffic with a display drawn on top, compare pairs of
> videos and answer questions. It takes about 30 minutes.
>
> Requirements:
> 1. Use a desktop or laptop computer with a screen of at least 1280 × 720 pixels (no phone or tablet) and a recent
>    Chrome, Firefox, Edge or Safari browser.
> 2. You must be 18 or older.
> 3. Take part only once. Repeated participation is not paid.
> 4. Stay on the study page until the end: do not switch tabs. Each video pair must be watched before you can answer.
>
> Steps:
> 1. Open the link: https://[app].herokuapp.com/?platform=microworkers
> 2. Read the information page and give consent, then follow the instructions.
> 3. At the end you get a completion code (14 characters, starting with AV). Copy it.
> 4. Paste the code below as proof.
>
> Some questions check whether you read carefully. If you answer too many of them wrongly the study ends early; you
> still get a code, paste it below for a partial payment.

## Required proof

> The completion code shown at the end of the study (for example AV7KQ2M9XH4PRT).

## Rating submissions

1. Copy the codes submitted on Microworkers into `codes.txt`, one per line (keep it with the study data, not in git).
2. Export the data from MongoDB (`download_data` in `config`, or `mongoexport`, see the README) and run:

   ```bash
   uv run python payments.py --data data/heroku_0.json --check $(cat codes.txt)
   ```

3. Per code: `full` → satisfied; `partial` → partial payment; `none` (not in the data, or left before the end) →
   not satisfied, with the reason "completion code not found in the study data".
4. A non-empty `review` (fast answers, same IP address as an earlier participant, rows from several sessions,
   finished faster than possible, incomplete data) is not refused automatically: look at the record and decide.
5. After payment is complete, delete `payments.csv` and `codes.txt` (they link codes to payment).

## Before opening the pilot

* the study runs at `https://[app].herokuapp.com/?debug=true` and a full test session (without `debug`) arrives in
  MongoDB and shows up as `full` in `payments.py --check`;
* `study_id` set to the pilot one and `make_design.py` run, so pilot participants are never pooled with the main study;
* consent text and amounts filled in (`public/js/study-text.js`).
