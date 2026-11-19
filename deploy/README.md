# Running the study on Heroku

The study runs as [bazilinskyy/ehmi-vehicles](https://github.com/bazilinskyy/ehmi-vehicles) does: a Node.js app on
Heroku (`Procfile`, `app.js`) that stores the responses in the `entries` collection of a MongoDB database (MongoDB
Atlas). One difference: the video clips. They are third-party footage, so they never go to GitHub; `heroku.sh` pushes
the committed code plus only the clips the design needs (197 clips, about 260 MB; Heroku allows 500 MB) straight to the
Heroku app's own git, and `app.js` serves them.

## Once

1. A MongoDB database (MongoDB Atlas, EU region), with a user that can write to it. Allow connections from anywhere
   (Heroku's addresses change).
2. A Heroku app with the Node.js buildpack (Node 22 from `package.json`), and its connection string:

   ```bash
   heroku create <app> --region eu
   heroku config:set CONNECTION='<MongoDB connection string>' -a <app>
   ```

   `heroku.sh` sets `SESSION_SECRET` (the key that signs the session tokens) on the first deploy. The app refuses to
   start on Heroku without it, since Heroku's disk is wiped at every restart.
3. For the analysis, put the same connection string in `secret` (`mongodb_uri`, from `default.secret`) and install
   `mongoexport` (MongoDB Database Tools).

## Every deploy

Commit first (the deploy takes `HEAD`), render the clips the design needs (`render_stimuli.py`), then:

```bash
DRY_RUN=1 deploy/heroku.sh <app>    # lists missing clips and the build size, pushes nothing
deploy/heroku.sh <app>
```

Open `https://<app>.herokuapp.com/?debug=true` to try it; debug sessions are marked and left out of the analysis.
Use at least one paid dyno during data collection (free/eco dynos sleep, and the first participant after a pause then
waits for the app to start).

## During and after data collection

* Export the data: set `download_data` to `true` in `config` and run the analysis (it writes `data/heroku_0.json`), or
  by hand: `mongoexport --uri <mongodb_uri> --collection entries --type json --out data/heroku_0.json`.
* Payments: `uv run python payments.py --data data/heroku_0.json --check <codes>`.
* When collection and payment are finished: scale the app down (`heroku ps:scale web=0 -a <app>`), export once more,
  replace the IP addresses in the export (`uv run python scrub_ips.py --data data/heroku_0.json --in-place`), archive
  it, and then empty the database (drop the `entries` collection) so the raw addresses are gone there too.
