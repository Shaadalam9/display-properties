// Server for the crowdsourced study, set up as in bazilinskyy/ehmi-vehicles:
// Express serves the jsPsych experiment and stores every batch of trial rows the
// browser sends as one document in the `entries` collection of a MongoDB database.
//
//   CONNECTION='<MongoDB connection string>' npm start      -> http://localhost:3000
//
// The live study runs on Heroku with a MongoDB database, as ehmi-vehicles does (deploy/README.md):
// set CONNECTION and SESSION_SECRET as config variables; the Procfile starts this file, and this
// server also serves the video clips from public/stimuli/ (deploy/heroku.sh puts them there).
// Without CONNECTION the rows are appended to a JSON lines file instead (DATA_FILE, default
// data/entries.jsonl), in the same shape as a mongoexport of `entries`: for running it locally
// and for the tests.
//
// Each stored batch carries the participant's IP address (`ip`), only to find people who take
// part more than once; the analysis keeps each address's first participant and replaces the
// addresses with placeholders, so no output holds them (avprops/crowd.py). The server keeps
// no request log.
//
// Session token: the page asks GET /session for a token when it opens (a random ID and the
// server's time, signed with a secret only the server knows) and sends it with every save. A
// save without a valid token is refused, and each batch is stored with the session ID, the
// time the session started and the time the batch arrived, all by the server's clock. So rows
// cannot be stored without opening the study, the analysis can tell how long a session really
// took (a completion faked by sending rows quickly is flagged), and rows of one code that come
// from more than one session are flagged too. The token holds nothing about the participant.

const crypto = require('crypto');
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

async function storeRows(doc) {
    if (Entry) {
        await Entry.create(doc);
    } else {
        await fs.promises.appendFile(LOCAL_FILE, JSON.stringify(doc) + '\n');
    }
}

// --- SESSION TOKENS
// The secret: SESSION_SECRET, or (locally) one made on the first start and kept next to the data
// file, so a restart of the server does not invalidate the sessions under way. Heroku's disk is
// wiped at every restart (at least daily), so there SESSION_SECRET is required.
function sessionSecret() {
    if (process.env.SESSION_SECRET) return process.env.SESSION_SECRET;
    if (process.env.DYNO) {
        console.error('SESSION_SECRET is not set: heroku config:set SESSION_SECRET=$(openssl rand -hex 32)');
        process.exit(1);
    }
    const file = path.join(path.dirname(LOCAL_FILE), 'session.secret');
    fs.mkdirSync(path.dirname(file), { recursive: true });
    if (!fs.existsSync(file)) {
        fs.writeFileSync(file, crypto.randomBytes(32).toString('hex'), { mode: 0o600 });
    }
    return fs.readFileSync(file, 'utf8').trim();
}
const SECRET = sessionSecret();
const SESSION_MAX_MS = 24 * 3600 * 1000;  // a token older than this is refused

const sign = (text) => crypto.createHmac('sha256', SECRET).update(text).digest('hex').slice(0, 32);

function newToken() {
    const id = crypto.randomBytes(12).toString('hex');
    const started = Date.now();
    return `${id}.${started}.${sign(`${id}.${started}`)}`;
}

/** The session ID and start time of a valid token, or null. */
function checkToken(token) {
    const parts = String(token || '').split('.');
    if (parts.length !== 3 || !/^[0-9a-f]{24}$/.test(parts[0]) || !/^\d{13}$/.test(parts[1])) return null;
    const expected = Buffer.from(sign(`${parts[0]}.${parts[1]}`));
    const given = Buffer.from(parts[2]);
    if (given.length !== expected.length || !crypto.timingSafeEqual(given, expected)) return null;
    const started = Number(parts[1]);
    const age = Date.now() - started;
    if (age < 0 || age > SESSION_MAX_MS) return null;
    return { id: parts[0], started: started };
}

// --- APP
const app = express();
// the participant's address comes from X-Forwarded-For, set by Heroku's router (one hop; DYNO is
// set there); locally only a proxy on the same machine is trusted
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

app.get('/session', (request, response) => {
    response.set('Cache-Control', 'no-store').json({ token: newToken() });
});

app.post('/experiment-data', async (request, response) => {
    const rows = request.body;
    if (!Array.isArray(rows) || rows.length === 0) {
        response.status(400).end();
        return;
    }
    const session = checkToken(request.get('X-Session'));
    if (!session) {
        response.status(403).end();
        return;
    }
    try {
        await storeRows({ data: rows, ip: request.ip, session: session.id, started: session.started,
                          received: Date.now() });
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
