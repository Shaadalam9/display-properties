# Running the study on a server of our own

The default deployment is Heroku with MongoDB, as in ehmi-vehicles (`deploy/heroku.sh`, see the main README).
This is the alternative: one virtual server (e.g. TU/e datacenter or SURF Research Cloud) that also stores the
responses itself.

The study runs on one TU/e-managed virtual server (TU/e datacenter or SURF Research Cloud), so
that no participant data passes through an outside hosting, database or video service:

* the Node.js server (`app.js`) serves the experiment and the video stimuli;
* responses are appended to one JSON lines file on TU/e storage (`DATA_FILE`); no database;
* nginx in front gives HTTPS and keeps no access log; the participant's IP address is stored
  only with the responses, to find duplicate participants, and the analysis masks it.

## What the server needs

* Linux with Node.js 22 and nginx
* about 5 GB of disk for the code and the video stimuli (`public/stimuli/`, ~3 GB), plus the
  responses (a few MB)
* a public HTTPS address that participants anywhere can reach, with a certificate
* outgoing bandwidth: each participant loads about 100 MB of video, so 400 participants are
  about 40 GB over the weeks of data collection
* a folder on TU/e storage that is backed up, for `DATA_FILE`

## Setup

```bash
sudo useradd --system --create-home study
sudo mkdir -p /opt/display-properties /srv/study-data && sudo chown study /opt/display-properties /srv/study-data
sudo -u study git clone https://github.com/Shaadalam9/display-properties.git /opt/display-properties
cd /opt/display-properties && sudo -u study npm ci --omit=dev
# the stimuli are not in git: copy them from the machine that rendered them
rsync -a public/stimuli/ study@SERVER:/opt/display-properties/public/stimuli/
sudo cp deploy/display-properties.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now display-properties
sudo cp deploy/nginx.conf /etc/nginx/sites-available/display-properties   # set server_name, certificate
sudo ln -s /etc/nginx/sites-available/display-properties /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

Keep `stimulus_url` at `"stimuli/"` in `default.config` (then `make_design.py`): the clips are
served by the same server. Open `https://<address>/?debug=true` to try it.

## During and after data collection

* `DATA_FILE` grows by one line per saved batch of trials; copy it to the project folder on TU/e
  storage regularly (it is already there if `DATA_FILE` points to that storage).
* Analyse with `uv run python analysis.py --data /srv/study-data/entries.jsonl`.
* When collection ends, stop the service (`sudo systemctl disable --now display-properties`)
  and remove the stimuli and the data file from the server once they are archived.
