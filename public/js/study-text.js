// Participant-facing text. Replace the placeholders with the ethics-approved wording
// before the pilot. Strings may contain HTML.
var STUDY_TEXT = {
    welcome:
        '<h1>Welcome</h1>' +
        '<p>Thank you for taking part.</p>' +
        '<p>In this study you will watch short videos recorded from a car that drives itself. The videos show a ' +
        'display that explains what the car is doing. You will tell us which parts of such a display matter to you.</p>' +
        '<p>The study takes about 25 minutes. Please stay on this page until the end and do not switch to other tabs.</p>',
    consent:
        '<h1>Information and consent</h1>' +
        '<p>[Information sheet: researchers, purpose, data handling, right to withdraw — to be added after ethics approval.]</p>',
    consentQuestion: 'I have read the information above and agree to take part.',
    declined: '<h1>Thank you</h1><p>You chose not to take part. You may now close this tab.</p>',
    comprehensionIntro:
        '<h1>What happens next?</h1>' +
        '<p>You will see a few short videos recorded from a car that drives itself. Each video stops before the end.</p>' +
        '<p>After each video, choose what you think the car will do next.</p>',
    baselineIntro: 'Watch this video of a car that drives itself. The coloured overlay is the car\'s display.',
    openQuestion: 'What would you want this display to show, or show differently? Write anything that comes to mind.',
    bwsIntro:
        '<h1>Part 1: what matters most</h1>' +
        '<p>Next you will see groups of display properties.</p>' +
        '<p>In each group, choose the property that is <strong>most</strong> important for you to understand what ' +
        'the car is doing and why, and the one that is <strong>least</strong> important.</p>' +
        '<p>Each property has a short example video: the left side shows one end of it, the right side the other.</p>',
    abIntro:
        '<h1>Part 2: comparing videos</h1>' +
        '<p>Next you will see pairs of videos, A and B, of the same situation.</p>' +
        '<p>The displays in the two videos may differ in one property. Watch both, then answer the questions below them.</p>',
    finalComment: 'Is there anything else you would like to tell us about the display? (optional)',
    tooSmall: 'Your screen is too small for this study. Please open the link on a computer with a screen of at least 1280 by 720 pixels.',
    mobile: 'This study cannot be completed on a phone or tablet. Please open the link on a computer.'
};

// Attention checks, as in ehmi-vehicles: shown between trials in random order; the study
// ends when attention_fail_at answers are wrong (setting in the config / design.json).
var ATTENTION_QUESTIONS = [
    { question: 'What is 2+2?', options: ['22', '4', '12'], correct: '4' },
    { question: 'What are you doing in this study?', options: ['Watching videos of a car that drives itself', 'Listening to music', 'Drawing patterns on the screen'], correct: 'Watching videos of a car that drives itself' },
    { question: 'Select the true statement.', options: ['Paris is the capital of Germany', 'Water is wet', 'The sun is purple'], correct: 'Water is wet' },
    { question: 'What do the videos in this study show?', options: ['Traffic seen from a car', 'Paintings', 'Supermarket products'], correct: 'Traffic seen from a car' },
    { question: 'Please select "Bus" to confirm you are reading carefully.', options: ['Car', 'Truck', 'Bus'], correct: 'Bus' }
];
