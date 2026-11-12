// Experiment for the AV display property study, built with jsPsych 8 in the same way
// as bazilinskyy/ehmi-vehicles: URL parameters, a worker code, full screen and a device
// check, attention checks that end the study after too many mistakes, and data sent to
// the server after every trial.
//
// Timeline: preload -> full screen -> device check -> welcome and consent -> demographics
// -> questionnaires -> comprehension clips ("what will the car do next?") -> baseline video
// + open question (Task C) -> Best-Worst Scaling (Task A, MaxDiff) -> A/B video pairs
// (Task B: each property in two scenes, catch pairs and one display vs no display pair) -> final
// comment -> finish page with the worker code.
//
// URL parameters:
//   debug=true          log to the console, skip the device check, allow continuing
//                       before a video ends, and mark every row debug = true
//   save_data=false     do not send data to the server
//   skip_to=tasks       with debug=true: start at the ranking and A/B tasks (for testing)
//   worker_code=<code>  code of the participant (PROLIFIC_PID is used when present;
//                       otherwise a random one is made and shown on the finish page)
//   platform=<name>     recruitment platform, stored with the data (e.g. microworkers)
//
// On Microworkers the link carries no worker ID: each participant gets a random code, shows
// it as the completion code at the end (also after failing the attention checks, for the
// partial payment) and pastes it into Microworkers as proof. payments.py lists every code
// with what it earns, so the study's data never holds the Microworkers ID.
'use strict';

var SAVE_DATA = true;  // save data or not

function getUrlParameter(name) {
    return new URLSearchParams(window.location.search).get(name);
}

var debug = /^true$/i.test(getUrlParameter('debug') || '');
var save_data = getUrlParameter('save_data') === null ? SAVE_DATA : /^true$/i.test(getUrlParameter('save_data'));

/**
 * Code of the participant: worker_code or PROLIFIC_PID from the URL, or a new random one
 * that cannot be guessed (it is the completion code that earns the payment).
 */
function getWorkerCode() {
    var code = getUrlParameter('worker_code') || getUrlParameter('PROLIFIC_PID');
    if (code && /^[A-Za-z0-9_-]{1,64}$/.test(code)) {
        return code;
    }
    var alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';  // no 0/O, 1/I: easy to copy by hand
    var bytes = new Uint8Array(12);
    window.crypto.getRandomValues(bytes);
    return 'AV' + Array.from(bytes, function(b) { return alphabet[b % alphabet.length]; }).join('');
}
var worker_code = getWorkerCode();
if (debug) {
    console.log('worker_code', worker_code, 'save_data', save_data);
}

var excluded = false;  // set when the participant cannot or does not continue
var blurred_trials = new Set();  // trial indices during which the tab or window lost focus

var jsPsych = initJsPsych({
    on_finish: function() {
        if (excluded) return;  // keep the message on screen
        window.location = 'finish?work=' + encodeURIComponent(worker_code);
    },
    on_interaction_data_update: function(data) {
        if (data.event === 'blur') blurred_trials.add(data.trial);
    }
});

// Same participant code, same order of trials and stimuli.
jsPsych.randomization.setSeed(worker_code);

var last_saved_index = -1;  // trial_index of the last row sent to the server

/**
 * Save data. Sends only rows not sent before. The optional callback runs once the data is
 * stored, or after the last failed attempt, so the study never waits forever on a save.
 */
function saveData(on_done) {
    var done = on_done || function() {};
    if (!save_data) {
        done();
        return;
    }
    var new_rows = jsPsych.data.get().filterCustom(function(row) { return row.trial_index > last_saved_index; });
    if (new_rows.count() === 0) {
        done();
        return;
    }
    var first_index = new_rows.values()[0].trial_index;
    // the next save sends only newer rows, unless this one fails (see below)
    last_saved_index = new_rows.last(1).values()[0].trial_index;
    var body = new_rows.json();
    var send = function(attempt) {
        // keepalive lets the request finish when the page is left meanwhile; browsers allow it
        // only for small bodies (64 kB)
        fetch('experiment-data', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body,
                                   keepalive: new Blob([body]).size < 60000 })
            .then(function(response) {
                if (!response.ok) throw new Error('HTTP ' + response.status);
                if (debug) console.log('data saved', new_rows.count(), 'rows');
                done();
            })
            .catch(function(err) {
                if (attempt < 3) {
                    setTimeout(function() { send(attempt + 1); }, 2000 * attempt);
                } else {
                    // send these rows again with the next save (the analysis keeps one copy of a row)
                    last_saved_index = Math.min(last_saved_index, first_index - 1);
                    alert('A problem occurred while saving your answers. Please contact the researcher.');
                    done();
                }
            });
    };
    send(1);
}

var save_data_block = {
    type: jsPsychCallFunction,
    func: function() { saveData(); }
};

/** Mark whether the participant left the page during this trial. */
function markHidden(data) {
    data.hidden = blurred_trials.has(data.trial_index);
}

function videoTag(src) {
    return '<div class="video-wrap"><video src="' + src + '" autoplay muted loop playsinline controls></video></div>';
}

/** In debug mode, name a missing clip on screen instead of showing a black box. */
function flagMissingVideos() {
    document.querySelectorAll('video').forEach(function(video) {
        var source = video.querySelector('source');
        // the error fires on <source> when the video uses one (video-button-response does)
        (source || video).addEventListener('error', function() {
            var note = document.createElement('div');
            note.className = 'video-missing';
            note.textContent = debug ? 'Clip not available yet: ' + (source || video).getAttribute('src')
                : 'This video could not be loaded. Please reload the page.';
            video.parentNode.style.position = 'relative';
            video.parentNode.appendChild(note);
        });
    });
}

function scaleHtml(name, points, low, high) {
    var html = '<div class="scale"><span class="anchor">' + low + '</span>';
    for (var i = 1; i <= points; i++) {
        html += '<label><input type="radio" name="' + name + '" value="' + i + '"' + (i === 1 ? ' required' : '') + '>' + i + '</label>';
    }
    return html + '<span class="anchor">' + high + '</span></div>';
}

function choiceHtml(name, options) {
    return '<div class="scale">' + options.map(function(o, i) {
        return '<label><input type="radio" name="' + name + '" value="' + o[0] + '"' + (i === 0 ? ' required' : '') + '>' + o[1] + '</label>';
    }).join('') + '</div>';
}

/**
 * Build the timeline from design.json (written by make_design.py).
 */
function buildTimeline(design) {
    jsPsych.data.addProperties({
        worker_code: worker_code,
        study_id: design.study_id,
        design_generated_at: design.generated_at,
        debug: debug,
        platform: getUrlParameter('platform') || (getUrlParameter('PROLIFIC_PID') ? 'prolific' : ''),
        prolific_study_id: getUrlParameter('STUDY_ID') || '',
        prolific_session_id: getUrlParameter('SESSION_ID') || ''
    });

    var props = {};
    design.properties.forEach(function(p) { props[p.id] = p; });
    var levelFile = function(prop, level, variant) {
        return prop + '_' + level + (level === 'high' && variant ? '_' + variant : '');
    };

    // --- per-participant assignment: one of the plans made in Python (design.plans, see
    // avprops/design.py build_plans), picked from the participant's code, so the experiment
    // and simulate.py share every random assignment. A plan holds the BWS version, the A/B
    // pairs in order (each tunable property in two scenes, categorical ones in one, catch
    // pairs of one clip on both sides, display vs no display; sides and palette variants
    // drawn) and the comprehension clips. It is logged in the first instructions row.
    var LEVELS = ['low', 'high', 'none', 'baseline'];
    var plan_index = Math.floor(Math.random() * design.plans.length);
    var plan = design.plans[plan_index];
    var variantName = function(k, v) { return k >= 0 && v >= 0 ? design.properties[k].variants[v] : null; };
    var bws_version = plan.bws_version;
    var bws_trials = design.bws.versions[bws_version];
    var pair_plan = plan.pairs.map(function(p) {
        return { property: p[0] < 0 ? 'no_display' : design.properties[p[0]].id, scene: design.scenes[p[1]].id,
                 a_level: LEVELS[p[2]], b_level: LEVELS[p[3]], high_variant: variantName(p[0], p[4]), catch: !!p[5] };
    });

    // stimulus_url (config): stimuli/ when the server has the clips, or the URL where they are hosted
    var clip = function(scene, name) { return design.stimulus_url + scene + '/' + name + '.mp4'; };
    var pairClip = function(plan, level) {
        return plan.property === 'no_display' ? clip(plan.scene, level)
            : clip(plan.scene, levelFile(plan.property, level, plan.high_variant));
    };
    var baseline_clip = clip(design.baseline_scene, 'baseline');

    // comprehension: one clip per scene with a question, at the plan's level of one property
    // that visibly differs in that scene; the clip stops before the car acts
    var comprehension_plan = design.comprehension.map(function(c, k) {
        var drawn = plan.comprehension[k];
        return { scene: c.scene, property: design.properties[drawn[0]].id, level: LEVELS[drawn[1]], spec: c,
                 high_variant: variantName(drawn[0], drawn[2]), options: jsPsych.randomization.shuffle(c.options) };
    });

    var timeline = [];

    // --- preload, full screen, device check
    var videos = [baseline_clip];
    pair_plan.forEach(function(p) { videos.push(pairClip(p, p.a_level), pairClip(p, p.b_level)); });
    comprehension_plan.forEach(function(c) { videos.push(clip(c.scene, levelFile(c.property, c.level, c.high_variant))); });
    design.properties.forEach(function(p) { if (p.preview) videos.push(design.stimulus_url + p.preview); });
    timeline.push({
        type: jsPsychPreload,
        video: videos,
        show_progress_bar: true,
        continue_after_error: true,
        max_load_time: 300000,
        data: { task: 'preload' }
    });
    timeline.push({ type: jsPsychFullscreen, fullscreen_mode: true });
    if (!debug) {
        timeline.push({
            type: jsPsychBrowserCheck,
            // only what the device check needs (as in the DMP: browser type and screen size)
            features: ['width', 'height', 'browser', 'mobile'],
            minimum_width: 1280,
            minimum_height: 720,
            inclusion_function: function(data) { return !data.mobile; },
            exclusion_message: function(data) {
                excluded = true;
                return '<h1>Unable to participate</h1><p>' + (data.mobile ? STUDY_TEXT.mobile : STUDY_TEXT.tooSmall) + '</p>';
            },
            data: { task: 'browser_check' }
        });
    }

    // --- welcome and consent
    timeline.push({
        type: jsPsychInstructions,
        pages: [STUDY_TEXT.welcome, STUDY_TEXT.consent],
        show_clickable_nav: true,
        data: { task: 'instructions', plan_index: plan_index, bws_version: bws_version, ab_plan: pair_plan,
                comprehension_plan: comprehension_plan.map(function(c) {
                    return { scene: c.scene, property: c.property, level: c.level, high_variant: c.high_variant };
                }),
                window_width_init: window.innerWidth, window_height_init: window.innerHeight }
    });
    timeline.push({
        type: jsPsychSurveyMultiChoice,
        questions: [{ prompt: STUDY_TEXT.consentQuestion, name: 'consent', options: ['Yes', 'No'], required: true }],
        data: { task: 'consent' },
        on_finish: function(data) {
            if (data.response.consent !== 'Yes') {
                excluded = true;
                saveData(function() { jsPsych.abortExperiment(STUDY_TEXT.declined); });
            }
        }
    });

    // --- demographics
    timeline.push({
        type: jsPsychSurveyText,
        preamble: '<h1>About you</h1>',
        questions: [
            { prompt: 'Age', name: 'age', required: true }
        ],
        data: { task: 'demographics' },
        on_load: function() {
            // native form validation: the browser blocks submitting until the age is valid
            var age = document.querySelector('input[data-name="age"]');
            age.type = 'number';
            age.min = 18;
            age.max = 99;
            age.step = 1;
        }
    });
    // countries as ISO codes (countries.js), so the analysis can match them with each scene's country
    var countrySelect = function(name) {
        return '<select name="' + name + '" required><option value="">Choose…</option>' +
            COUNTRIES.map(function(c) { return '<option value="' + c[0] + '">' + c[1] + '</option>'; }).join('') +
            '</select>';
    };
    timeline.push({
        type: jsPsychSurveyHtmlForm,
        preamble: '<h1>About you</h1>',
        html: '<div class="questions"><p class="q">Nationality</p>' + countrySelect('nationality') +
            '<p class="q">Country you currently live in</p>' + countrySelect('country_residence') + '</div>',
        button_label: 'Continue',
        data: { task: 'demographics' }
    });
    timeline.push({
        type: jsPsychSurveyMultiChoice,
        preamble: '<h1>About you</h1>',
        questions: [
            { prompt: 'Gender', name: 'gender', options: ['Female', 'Male', 'Other', 'Prefer not to say'], required: true },
            { prompt: 'Do you have a driving licence?', name: 'licence', options: ['Yes', 'No'], required: true },
            { prompt: 'How often do you travel by car (as driver or passenger)?', name: 'car_frequency', options: ['Daily', 'A few times a week', 'A few times a month', 'Rarely or never'], required: true },
            { prompt: 'How much experience do you have with driver-assistance or self-driving features?', name: 'automation_experience', options: ['None', 'A little', 'Some', 'A lot'], required: true },
            { prompt: 'Have you ever ridden in a car that drives itself?', name: 'av_ride', options: ['Yes', 'No'], required: true },
            { prompt: 'How familiar are you with head-up or augmented-reality displays in cars?', name: 'ar_familiarity', options: ['Not at all', 'Slightly', 'Moderately', 'Very'], required: true },
            { prompt: 'Do you have any form of colour vision deficiency (colour blindness)?', name: 'colour_vision_deficiency', options: ['Yes', 'No', 'Not sure', 'Prefer not to say'], required: true }
        ],
        data: { task: 'demographics' }
    });

    // --- questionnaires (study/questionnaires.json); answers are stored as 1..n
    design.questionnaires.forEach(function(q) {
        timeline.push({
            type: jsPsychSurveyLikert,
            preamble: '<p>' + q.preamble + '</p>',
            questions: q.items.map(function(item) {
                return { prompt: item.text, name: item.id, labels: q.labels, required: true };
            }),
            data: { task: 'questionnaire', questionnaire: q.id }
        });
    });
    timeline.push(save_data_block);

    // --- comprehension: what will the car do next? (before the participant sees the
    // baseline clip or any property names, so nothing tells them the answer)
    if (comprehension_plan.length) {
        timeline.push({ type: jsPsychInstructions, pages: [STUDY_TEXT.comprehensionIntro], show_clickable_nav: true });
    }
    comprehension_plan.forEach(function(c, k) {
        timeline.push({
            type: jsPsychVideoButtonResponse,
            stimulus: [clip(c.scene, levelFile(c.property, c.level, c.high_variant))],
            prompt: '<p><strong>' + c.spec.question + '</strong></p>',
            choices: c.options,
            stop: c.spec.stop_s,
            response_allowed_while_playing: debug,
            width: 960,
            data: { task: 'comprehension', index: k, scene: c.scene, property: c.property, level: c.level,
                    high_variant: c.high_variant,
                    options: c.options, expected: c.spec.correct },
            on_load: flagMissingVideos,
            on_finish: function(data) {
                data.answer = c.options[data.response];
                data.correct = data.answer === c.spec.correct;
                markHidden(data);
            }
        });
        timeline.push(save_data_block);
    });

    // --- Task C: baseline video and open question
    timeline.push({
        type: jsPsychVideoButtonResponse,
        stimulus: [baseline_clip],
        prompt: '<p>' + STUDY_TEXT.baselineIntro + '</p>',
        choices: ['Continue'],
        response_allowed_while_playing: debug,
        width: 960,
        data: { task: 'baseline', scene: design.baseline_scene },
        on_load: flagMissingVideos,
        on_finish: markHidden
    });
    timeline.push({
        type: jsPsychSurveyText,
        questions: [{ prompt: STUDY_TEXT.openQuestion, name: 'open', rows: 6, columns: 80, required: true }],
        data: { task: 'open_question', scene: design.baseline_scene },
        on_finish: markHidden
    });
    timeline.push(save_data_block);
    var tasks_start = timeline.length;

    // --- Task A and Task B trials, with the attention checks between them
    var bws_blocks = bws_trials.map(function(items, t) {
        var alternatives = items.map(function(id) {
            var p = props[id];
            // a short looping example of the property (render_stimuli.py --previews): low on the
            // left, high on the right (the palettes: all four)
            var preview = p.preview ? '<video class="preview" src="' + design.stimulus_url + p.preview +
                '" autoplay muted loop playsinline></video>' : '';
            return '<div class="alt">' + preview + '<div><strong>' + p.name + '</strong><br>' + p.description +
                '<br><span class="levels">' + p.low + ' &hellip; ' + p.high + '</span></div></div>';
        });
        return {
            type: jsPsychMaxdiff,
            preamble: '<h2>Which matters most? (' + (t + 1) + ' of ' + bws_trials.length + ')</h2>' +
                '<p>For understanding what the car is doing and why.</p>',
            alternatives: alternatives,
            labels: ['Most important', 'Least important'],
            required: true,
            data: { task: 'bws', trial: t, items: items },
            on_finish: function(data) {
                data.best = items[alternatives.indexOf(data.response.left)];
                data.worst = items[alternatives.indexOf(data.response.right)];
                markHidden(data);
            }
        };
    });

    var points = design.ab.scale_points;
    var ab_blocks = pair_plan.map(function(plan, i) {
        var html =
            '<h2>Compare the videos (' + (i + 1) + ' of ' + pair_plan.length + ')</h2>' +
            '<div class="pair"><div><h3>Video A</h3>' + videoTag(pairClip(plan, plan.a_level)) + '</div>' +
            '<div><h3>Video B</h3>' + videoTag(pairClip(plan, plan.b_level)) + '</div></div>' +
            '<div class="questions">' +
            '<p class="q">Did you notice a difference between the displays in A and B?</p>' + choiceHtml('noticed', [['yes', 'Yes'], ['no', 'No']]) +
            '<p class="q">Which display would you prefer to have while riding in a car that drives itself?</p>' +
            choiceHtml('preferred', [['-2', 'A, clearly'], ['-1', 'A, slightly'], ['0', 'No preference'],
                                     ['1', 'B, slightly'], ['2', 'B, clearly']]) +
            '<p class="q">How much does the difference between A and B matter to you?</p>' + scaleHtml('importance', points, 'Not at all', 'Very much');
        if (design.ab.collect_ratings) {
            var items = [['understanding', 'I understood what the car was doing and why.'],
                         ['safety', 'The car felt safe.'],
                         ['load', 'The display was distracting or hard to process.']];
            html += '<div class="ratings">' + ['A', 'B'].map(function(option) {
                return '<div><h3>Video ' + option + '</h3>' + items.map(function(item) {
                    return '<p class="q">' + item[1] + '</p>' + scaleHtml(option + '_' + item[0], points, 'Strongly disagree', 'Strongly agree');
                }).join('') + '</div>';
            }).join('') + '</div>';
        }
        html += '</div>';
        return {
            type: jsPsychSurveyHtmlForm,
            html: html,
            button_label: 'Continue',
            data: { task: 'ab', index: i, property: plan.property, scene: plan.scene, a_level: plan.a_level,
                    b_level: plan.b_level, high_variant: plan.high_variant, catch: !!plan.catch },
            on_load: function() {
                flagMissingVideos();
                // Continue only once the clips have played through once (not in debug mode)
                var button = document.querySelector('#jspsych-survey-html-form-next');
                if (!debug && button && design.ab.min_view_s > 0) {
                    button.disabled = true;
                    setTimeout(function() { button.disabled = false; }, design.ab.min_view_s * 1000);
                }
            },
            on_finish: function(data) {
                var r = data.response;
                data.noticed = r.noticed === 'yes';
                // -2 (A, clearly) .. 2 (B, clearly); preference: the same towards the high level
                // (towards the display, for the no-display pair), 0 = no preference
                var towards_b = Number(r.preferred);
                data.preferred_option = towards_b < 0 ? 'A' : (towards_b > 0 ? 'B' : 'none');
                data.preferred_level = towards_b < 0 ? plan.a_level : (towards_b > 0 ? plan.b_level : 'none');
                var b_up = plan.property === 'no_display' ? plan.b_level === 'baseline' : plan.b_level === 'high';
                data.preference = plan.catch ? null : (b_up ? towards_b : -towards_b);
                data.importance = Number(r.importance);
                if (design.ab.collect_ratings) {
                    var ratings = {};
                    [['A', plan.a_level], ['B', plan.b_level]].forEach(function(pair) {
                        ratings[pair[1]] = {
                            understanding: Number(r[pair[0] + '_understanding']),
                            safety: Number(r[pair[0] + '_safety']),
                            load: Number(r[pair[0] + '_load'])
                        };
                    });
                    data.ratings = ratings;
                }
                markHidden(data);
            }
        };
    });

    // attention checks go before randomly chosen trials of either part
    var attention_wrong = 0;
    var questions = jsPsych.randomization.shuffle(ATTENTION_QUESTIONS);
    var n_trials = bws_blocks.length + ab_blocks.length;
    var slots = jsPsych.randomization.shuffle(Array.from({ length: n_trials }, function(_, k) { return k; }))
        .slice(0, Math.min(questions.length, n_trials));
    var attentionBlock = function(q, k) {
        return {
            type: jsPsychSurveyMultiChoice,
            questions: [{ prompt: q.question, name: 'attention', options: q.options, required: true }],
            data: { task: 'attention', question: q.question, expected: q.correct },
            on_finish: function(data) {
                data.correct = data.response.attention === q.correct;
                if (!data.correct) {
                    attention_wrong++;
                    if (debug) console.log('wrong attention answer, total', attention_wrong);
                    if (attention_wrong >= design.attention_fail_at) {
                        excluded = true;
                        saveData(function() { window.location = 'finish_fail?work=' + encodeURIComponent(worker_code); });
                    }
                }
            }
        };
    };
    var pushTrial = function(block, k) {
        var slot = slots.indexOf(k);
        if (slot !== -1) timeline.push(attentionBlock(questions[slot], k));
        timeline.push(block);
        timeline.push(save_data_block);
    };

    timeline.push({ type: jsPsychInstructions, pages: [STUDY_TEXT.bwsIntro], show_clickable_nav: true });
    bws_blocks.forEach(function(block, k) { pushTrial(block, k); });
    timeline.push({ type: jsPsychInstructions, pages: [STUDY_TEXT.abIntro], show_clickable_nav: true });
    ab_blocks.forEach(function(block, k) { pushTrial(block, bws_blocks.length + k); });

    // --- end
    timeline.push({
        type: jsPsychSurveyText,
        questions: [{ prompt: STUDY_TEXT.finalComment, name: 'comment', rows: 5, columns: 80, required: false }],
        data: { task: 'final' }
    });
    // wait until the last rows are stored: the finish page follows right after, and leaving
    // the page cancels a request still on its way (the last row marks the study as finished)
    timeline.push({
        type: jsPsychCallFunction,
        async: true,
        func: function(finish) { saveData(function() { finish(); }); }
    });
    timeline.push({ type: jsPsychFullscreen, fullscreen_mode: false });

    if (debug) {
        console.log('BWS version', bws_version, 'A/B plan', pair_plan, 'comprehension', comprehension_plan,
                    'attention before trials', slots);
    }
    if (debug && getUrlParameter('skip_to') === 'tasks') {
        timeline.splice(2, tasks_start - 2);  // keep the preload and full screen
    }
    return timeline;
}

fetch('design.json', { cache: 'no-store' })
    .then(function(response) {
        if (!response.ok) throw new Error('design.json: HTTP ' + response.status);
        return response.json();
    })
    .then(function(design) { jsPsych.run(buildTimeline(design)); })
    .catch(function(err) {
        document.body.innerHTML = '<p>The study could not be loaded. Please reload the page.</p>';
        console.error(err);
    });
