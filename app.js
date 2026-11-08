// Server for the crowdsourced study, set up as in bazilinskyy/ehmi-vehicles:
// Express serves the jsPsych experiment and stores every batch of trial rows the
// browser sends as one document in the `entries` collection of a MongoDB database.
//
//   CONNECTION='<MongoDB connection string>' npm start      -> http://localhost:3000
//
// Without CONNECTION the rows are appended to a JSON lines file instead (DATA_FILE, default
// data/entries.jsonl), in the same shape as a mongoexport of `entries`. This is how the study
// runs on a TU/e server: DATA_FILE points to TU/e storage, the video stimuli are served from
// public/stimuli/ by this server too, and no data goes to an outside service (deploy/).
// On Heroku, set CONNECTION as a config variable; the Procfile starts this file.
//
// Each stored batch carries the participant's IP address (`ip`), only to find people who take
// part more than once; the analysis keeps each address's first participant and replaces the
// addresses with placeholders, so no output holds them (avprops/crowd.py). The server keeps
// no request log.

const fs = require('fs');
const path = require('path');
const express = require('express');
const mongoose = require('mongoose');

const ROOT = __dirname;
const PUBLIC = path.join(ROOT, 'public');
const LOCAL_FILE = path.resolve(ROOT, process.env.DATA_FILE || path.join('data', 'entries.jsonl'));

// jsPsych and the plugins come from npm (versions pinned in package-lock.json).
const JSPSYCH_PLUGINS = [
    'browser-check', 'call-function', 'fullscreen', 'instructions', 'maxdiff', 'preload',
    'survey-html-form', 'survey-likert', 'survey-multi-choice', 'survey-text', 'video-button-response',
];

// --- STORAGE
let Entry = null;
if (process.env.CONNECTION) {
    mongoose.connect(process.env.CONNECTION, { serverSelectionTimeoutMS: 30000 })
        .then(() => console.log('database opened'))
        .catch((err) => {
            console.error('connection error', err);
            process.exit(1);
        });
    Entry = mongoose.model('Entry', new mongoose.Schema({}, { strict: false }));
} else {
    console.log('CONNECTION is not set: storing data in ' + LOCAL_FILE);
    fs.mkdirSync(path.dirname(LOCAL_FILE), { recursive: true });
}

async function storeRows(rows, ip) {
    if (Entry) {
        await Entry.create({ data: rows, ip: ip });
    } else {
        await fs.promises.appendFile(LOCAL_FILE, JSON.stringify({ data: rows, ip: ip }) + '\n');
    }
}

// --- APP
const app = express();
// the participant's address comes from X-Forwarded-For, set by the proxy in front: on Heroku its
// router (one hop, DYNO is set there), on a server of our own nginx on the same machine
app.set('trust proxy', process.env.DYNO ? 1 : 'loopback');
app.use(express.json({ limit: '10mb' }));

const file = (relative) => (request, response) => response.sendFile(path.join(ROOT, relative));
app.get('/jspsych/jspsych.js', file('node_modules/jspsych/dist/index.browser.min.js'));
app.get('/jspsych/jspsych.css', file('node_modules/jspsych/css/jspsych.css'));
for (const name of JSPSYCH_PLUGINS) {
    app.get(`/jspsych/plugin-${name}.js`, file(`node_modules/@jspsych/plugin-${name}/dist/index.browser.min.js`));
}
// clips are cached by the browser, so a pair shown again (the repeats) is not downloaded twice;
// express.static answers range requests, which video players use to stream
app.use('/stimuli', express.static(path.join(PUBLIC, 'stimuli'), { index: false, maxAge: '1d', immutable: true }));
app.use(express.static(PUBLIC, { index: false }));

// --- ROUTING
const page = (name) => (request, response) => response.sendFile(path.join(PUBLIC, 'views', name));
app.get('/', page('experiment.html'));
app.get('/experiment', page('experiment.html'));
app.get('/finish', page('finish.html'));
app.get('/finish_fail', page('finish_fail.html'));

app.post('/experiment-data', async (request, response) => {
    const rows = request.body;
    if (!Array.isArray(rows) || rows.length === 0) {
        response.status(400).end();
        return;
    }
    try {
        await storeRows(rows, request.ip);
        response.end();
    } catch (err) {
        console.error('could not store data', err);
        response.status(500).end();
    }
});

// --- START THE SERVER
const server = app.listen(process.env.PORT || 3000, () => {
    console.log('Listening on port %d', server.address().port);
});
