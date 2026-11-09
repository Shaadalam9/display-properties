// Participant-facing text. The information sheet follows the data management plan; fill in
// the [placeholders] and match it to the ethics-approved wording before the pilot.
// Strings may contain HTML.
var STUDY_TEXT = {
    welcome:
        '<h1>Welcome</h1>' +
        '<p>Thank you for taking part.</p>' +
        '<p>In this study you will watch short videos of real traffic, filmed from cars driven by people. We have ' +
        'added a display on top of the videos, like one that a car that drives itself could show its passengers.</p>' +
        '<p>While you watch, <strong>imagine that you are a passenger and the car drives itself</strong>: the ' +
        'display explains what the car is doing and why. You will tell us which parts of such a display matter to you.</p>' +
        '<p>The study takes about 30 minutes. Please stay on this page until the end and do not switch to other tabs.</p>',
    consent:
        '<h1>Information and consent</h1>' +
        '<div class="info-sheet">' +
        '<p><strong>Who we are.</strong> This study is run by researchers at Eindhoven University of Technology ' +
        '(TU/e), the Netherlands [and partners, as in the approved application]. It has been approved by ' +
        '[ethics committee, reference number].</p>' +
        '<p><strong>Purpose.</strong> We want to find out which parts of a display in a future self-driving car ' +
        'would help passengers understand what the car does and why.</p>' +
        '<p><strong>What you will do.</strong> Answer a few questions about yourself, watch short videos of traffic ' +
        'and say which display features matter to you and which of two videos you prefer. The videos were filmed ' +
        'by people driving ordinary cars and are publicly available; the researchers added the display. You are ' +
        'asked to imagine that the car drives itself. ' +
        'This takes about 30 minutes. Some videos show busy traffic and pedestrians close to the car.</p>' +
        '<p><strong>Payment.</strong> You are paid through the crowdsourcing platform with the code shown at the end. ' +
        'If the study is ended early because of incorrect answers to the screening questions, you receive a ' +
        'partial payment.</p>' +
        '<p><strong>Your data.</strong> We record your answers, response times, age, gender, nationality, country of ' +
        'residence, driving experience, colour vision (you may choose "prefer not to say") and the type of browser ' +
        'and screen. We do not ask for your name, email address or exact location. Your IP address is stored ' +
        'with your answers only to detect people who take part more than once; it is then replaced by a ' +
        'placeholder and never published. During the study the data are stored in a secured online database; ' +
        'afterwards they are moved to TU/e storage and deleted from the database. An anonymised version of the data ' +
        '(without codes or IP addresses, with age in ranges) will be published in an open research repository ' +
        'and used in scientific publications.</p>' +
        '<p><strong>Voluntary.</strong> Taking part is voluntary. You can stop at any time by closing this page, ' +
        'without giving a reason. To have your data removed afterwards, send your completion code to ' +
        '[contact email] before [date]; after that, the data can no longer be linked to you.</p>' +
        '<p><strong>Questions or complaints.</strong> [contact email]; about how your data are handled: the TU/e ' +
        'data protection officer, [privacy contact].</p>' +
        '</div>' +
        '<p>By continuing you confirm that:</p>' +
        '<ul class="consent-list">' +
        '<li>you are 18 years or older;</li>' +
        '<li>you have read and understood the information above;</li>' +
        '<li>you agree that your answers are used as described, including publication of the anonymised data;</li>' +
        '<li>you take part voluntarily and know you can stop at any time.</li>' +
        '</ul>',
    consentQuestion: 'I am 18 years or older and agree to take part in this study as described above.',
    declined: '<h1>Thank you</h1><p>You chose not to take part. You may now close this tab.</p>',
    comprehensionIntro:
        '<h1>What happens next?</h1>' +
        '<p>You will see a few short videos with the display. Imagine that the car drives itself. Each video stops ' +
        'before the end.</p>' +
        '<p>After each video, choose what you think the car will do next.</p>',
    baselineIntro: 'Watch this video and imagine that the car drives itself. The coloured overlay is its display.',
    openQuestion: 'What would you want this display to show, or show differently? Write anything that comes to mind. ' +
        '(Please do not include your name or other details that could identify you.)',
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
    finalComment: 'Is there anything else you would like to tell us about the display? (optional; please do not ' +
        'include your name or other details that could identify you)',
    tooSmall: 'Your screen is too small for this study. Please open the link on a computer with a screen of at least 1280 by 720 pixels.',
    mobile: 'This study cannot be completed on a phone or tablet. Please open the link on a computer.'
};

// Attention checks, as in ehmi-vehicles: shown between trials in random order; the study
// ends when attention_fail_at answers are wrong (setting in the config / design.json).
var ATTENTION_QUESTIONS = [
    { question: 'What is 2+2?', options: ['22', '4', '12'], correct: '4' },
    { question: 'What are you doing in this study?', options: ['Watching videos of traffic with a display', 'Listening to music', 'Drawing patterns on the screen'], correct: 'Watching videos of traffic with a display' },
    { question: 'Select the true statement.', options: ['Paris is the capital of Germany', 'Water is wet', 'The sun is purple'], correct: 'Water is wet' },
    { question: 'What do the videos in this study show?', options: ['Traffic seen from a car', 'Paintings', 'Supermarket products'], correct: 'Traffic seen from a car' },
    { question: 'Please select "Bus" to confirm you are reading carefully.', options: ['Car', 'Truck', 'Bus'], correct: 'Bus' }
];
