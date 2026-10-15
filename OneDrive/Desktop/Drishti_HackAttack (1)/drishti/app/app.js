/* =====================================================================
   Drishti – AI eyes for the visually impaired
   ---------------------------------------------------------------------
   Modes
     GUARD     real-time obstacle detection with DIRECTIONAL SOUND:
               left zone  → low beep in the LEFT ear
               right zone → high beep in the RIGHT ear
               ahead      → double beep in BOTH ears
               faster beeps = closer. Plus vibration + spoken warnings.
     DESCRIBE  one-shot spoken summary with CLOCK-FACE directions
     READ      OCR of signs / labels / boards, read aloud (Eng/Hin/Kan)
     FIND      "find bottle": Geiger-counter guidance with distance,
               left/right, up/down and "within reach"
     WASHROOM  / EXIT: our OWN trained model (synthetic data) + OCR fallback
   Everything runs ON THE DEVICE: no images leave the phone.
   ===================================================================== */
'use strict';

// ------------------------------------------------------------------ config
const CFG = {
  detectEveryMs: 160,          // detection loop period
  minScore: 0.5,               // COCO-SSD confidence
  signEveryMs: 650,            // custom sign model period
  signThreshold: 0.85,         // confidence to announce a sign
  signStableFrames: 2,         // consecutive frames before announcing
  announceCooldownMs: 4500,    // don't repeat the same object too often
  vfovDeg: 60,                 // approx. camera vertical field of view
  hfovDeg: 64,                 // approx. camera horizontal field of view (radar)
  stepMeters: 0.7,
  leftZone: 0.42,              // x < 0.42 → "left"  (left ear only)
  rightZone: 0.58,             // x > 0.58 → "right" (right ear only); between = ahead (both ears)
  findCentre: 0.12,            // |x-0.5| < 0.12 → "straight ahead" in Find mode
  reachMeters: 0.75,           // closer than this → "within reach"
  radarRange: 5,               // metres shown on the radar
  // collision prediction
  approachRate: 0.3,           // object must grow ≥30 % per second to count as approaching
  ttcWarn: 3.5,                // warn when time-to-collision is below this (seconds)
  crossRate: 0.35,             // sideways speed (fraction of view per second) to count as crossing
};

// Objects that matter while walking (COCO-SSD class names)
const OBSTACLES = new Set(['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'chair', 'couch',
  'bench', 'potted plant', 'bed', 'dining table', 'toilet', 'dog', 'cat', 'suitcase', 'fire hydrant',
  'stop sign', 'parking meter', 'refrigerator', 'tv', 'backpack', 'umbrella', 'cow', 'horse', 'sink', 'oven']);

// Typical real-world heights (m) → distance estimate from box height (pinhole model)
const REAL_HEIGHT = { person: 1.65, chair: 0.9, couch: 0.85, bench: 0.8, 'dining table': 0.75, bed: 0.6,
  'potted plant': 0.6, bicycle: 1.0, motorcycle: 1.1, car: 1.5, bus: 3.0, truck: 3.0, dog: 0.5, cat: 0.3,
  suitcase: 0.6, backpack: 0.45, bottle: 0.25, cup: 0.1, 'wine glass': 0.18, 'cell phone': 0.15,
  laptop: 0.25, tv: 0.6, book: 0.24, keyboard: 0.04, mouse: 0.04, remote: 0.18, toilet: 0.75,
  refrigerator: 1.7, sink: 0.3, clock: 0.3, vase: 0.3, umbrella: 0.9, handbag: 0.3, 'teddy bear': 0.3,
  scissors: 0.18, toothbrush: 0.18, banana: 0.2, apple: 0.08, orange: 0.08, bowl: 0.08, 'fire hydrant': 0.8 };

// Spoken word → COCO class (for "find …")
const FIND_ALIASES = {
  bottle: 'bottle', 'water bottle': 'bottle', water: 'bottle', cup: 'cup', mug: 'cup', glass: 'wine glass',
  phone: 'cell phone', mobile: 'cell phone', 'cell phone': 'cell phone', laptop: 'laptop', computer: 'laptop',
  chair: 'chair', seat: 'chair', table: 'dining table', person: 'person', someone: 'person', people: 'person',
  bag: 'backpack', backpack: 'backpack', handbag: 'handbag', purse: 'handbag', book: 'book', remote: 'remote',
  keyboard: 'keyboard', mouse: 'mouse', tv: 'tv', television: 'tv', clock: 'clock', umbrella: 'umbrella',
  scissors: 'scissors', bowl: 'bowl', banana: 'banana', apple: 'apple', toothbrush: 'toothbrush',
  sofa: 'couch', couch: 'couch', bench: 'bench', bed: 'bed', door: null, keys: null, key: null, wallet: null,
};
const FIND_MENU = ['bottle', 'cup', 'phone', 'laptop', 'chair', 'person', 'bag', 'book', 'washroom', 'exit'];
const TRAFFIC_LABELS = ['pedestrian_crossing', 'road_work', 'school_ahead', 'speed_breaker', 'caution', 'no_entry', 'stop'];
const SIGN_TARGETS = { washroom: ['washroom_men', 'washroom_women'], toilet: ['washroom_men', 'washroom_women'],
  restroom: ['washroom_men', 'washroom_women'], bathroom: ['washroom_men', 'washroom_women'], exit: ['exit_sign'],
  traffic: TRAFFIC_LABELS };
// What to SAY for each sign label (labels must match the class names used in Teachable Machine)
const SIGN_SPEECH = {
  washroom_men: "Men's washroom sign", washroom_women: "Women's washroom sign", exit_sign: 'Exit sign',
  pedestrian_crossing: 'Pedestrian crossing sign. Zebra crossing ahead',
  road_work: 'Road work sign. Work in progress, be careful',
  school_ahead: 'School ahead sign. Watch for children',
  speed_breaker: 'Speed breaker sign. Uneven road ahead',
  caution: 'Caution sign. Danger ahead, go slowly',
  no_entry: 'No entry sign. Do not go this way',
  stop: 'Stop sign', washroom: 'Washroom sign',
};
const signName = l => SIGN_SPEECH[l] || cap(String(l).replace(/_/g, ' ')) + ' sign';

// ------------------------------------------------------------------ state
const S = {
  mode: 'idle',            // idle | guard | find | sign
  findTarget: null, findWord: '', signTargets: null,
  coco: null, signModel: null, signLabels: [], ocr: null, ocrLang: null,
  lastDetections: [], lastAnnounce: new Map(), signStreak: { label: null, n: 0 },
  lastSignAnnounce: new Map(), busyRead: false, haptics: true, showBoxes: true, verbose: false,
  rate: 1.1, voiceLang: 'en-IN', listening: false, mirror: false,
  track: null,             // Find-mode tracking state
  ocrScanBusy: false, lastOcrScan: 0, history: [],
  timers: [],              // pending sound-test timers (cleared by Stop)
  seen: new Map(),         // Guard: objects already announced → last time seen
  lastGuardSpeak: 0, lastUrgent: 0, lang: 'en',
  tracks: [],              // objects followed across frames (collision prediction)
  badgeHoldUntil: 0,
};

const $ = (id) => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), ctx = overlay.getContext('2d');
const work = document.createElement('canvas'); // scratch canvas for frames

// ================================================================ SPEECH OUT
// Rules: NEVER queue messages. A more important message cuts off a less important one;
// anything else that arrives while talking is simply dropped. Priority 3 = urgent / user asked.
let speakingPriority = 0;
function speak(text, { priority = 1, lang } = {}) {
  if (!('speechSynthesis' in window)) { setCaption(text); return; }
  const synth = window.speechSynthesis;
  if (synth.speaking || synth.pending) {
    if (priority > speakingPriority || priority >= 3) synth.cancel();
    else return; // busy talking: drop it, don't pile up
  }
  setCaption(text);
  const u = new SpeechSynthesisUtterance(text);
  u.rate = S.rate;
  u.lang = lang || detectScriptLang(text);
  const v = pickVoice(u.lang);
  if (v) u.voice = v;
  speakingPriority = priority;
  u.onend = u.onerror = () => { speakingPriority = 0; };
  synth.speak(u);
}
/** Silence everything immediately (speech + beeps + pending sound-test timers) */
function silence() {
  if ('speechSynthesis' in window) speechSynthesis.cancel();
  speakingPriority = 0;
  beeper.stop();
  S.timers.forEach(clearTimeout); S.timers = [];
}
function detectScriptLang(t) {
  if (/[ಀ-೿]/.test(t)) return 'kn-IN';   // Kannada
  if (/[ऀ-ॿ]/.test(t)) return 'hi-IN';   // (Hindi)
  return 'en-IN';
}
let voices = [];
function loadVoices() { voices = window.speechSynthesis ? speechSynthesis.getVoices() : []; }
if ('speechSynthesis' in window) { loadVoices(); speechSynthesis.onvoiceschanged = loadVoices; }
function pickVoice(lang) {
  return voices.find(v => v.lang === lang) || voices.find(v => v.lang.startsWith(lang.slice(0, 2))) || null;
}
function setCaption(t) {
  const cap = $('caption');
  if (cap.textContent && cap.textContent !== t && !/^Listening|^Starting|^Loading/.test(cap.textContent)) {
    S.history.unshift(cap.textContent);
    S.history = S.history.slice(0, 4);
    $('history').innerHTML = S.history.map(h => `<li>${h.replace(/[<&]/g, c => (c === '<' ? '&lt;' : '&amp;'))}</li>`).join('');
  }
  cap.textContent = t;
}

// ================================================================ SOUND (directional)
let actx = null;
function audio() {
  if (!actx) {
    actx = new (window.AudioContext || window.webkitAudioContext)();
    // force true 2-channel output, no automatic mixing between ears
    try { actx.destination.channelCount = 2; actx.destination.channelInterpretation = 'discrete'; } catch (_) {}
  }
  if (actx.state === 'suspended') actx.resume();
  return actx;
}
/** One tone. pan < 0 → LEFT ear ONLY, pan > 0 → RIGHT ear ONLY, 0 → both. `at` = delay in seconds. */
function tone({ freq = 880, dur = 0.08, pan = 0, vol = 0.4, type = 'sine', at = 0 } = {}) {
  const a = audio(), t = a.currentTime + at;
  const o = a.createOscillator(), g = a.createGain();
  o.type = type; o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(vol, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g);
  if (S.swapEars) pan = -pan;               // earbuds wired / paired the other way round
  // Route the sound into exactly one channel of a stereo merger (0 = left, 1 = right)
  const out = a.createChannelMerger(2);
  out.channelInterpretation = 'discrete';
  if (pan < 0) g.connect(out, 0, 0);
  else if (pan > 0) g.connect(out, 0, 1);
  else { g.connect(out, 0, 0); g.connect(out, 0, 1); }
  out.connect(a.destination);
  o.start(t); o.stop(t + dur + 0.03);
}
function zoneOf(x) { return x < CFG.leftZone ? 'left' : x > CFG.rightZone ? 'right' : 'ahead'; }
/**
 * Directional cue that works on headphones AND laptop/phone speakers:
 *  left  → fully in the LEFT channel and a LOWER pitch
 *  right → fully in the RIGHT channel and a HIGHER pitch
 *  ahead → a quick DOUBLE beep in both ears
 */
function cue(x, base = 880, type = 'sine') {
  const z = zoneOf(x);
  if (z === 'left') tone({ freq: base * 0.7, pan: -1, type });
  else if (z === 'right') tone({ freq: base * 1.35, pan: 1, type });
  else { tone({ freq: base, pan: 0, type, dur: 0.05 }); tone({ freq: base, pan: 0, type, dur: 0.05, at: 0.09 }); }
}
function chime() { tone({ freq: 660, dur: 0.12 }); tone({ freq: 990, dur: 0.2, at: 0.13 }); }
function vibrate(p) { if (S.haptics && navigator.vibrate) navigator.vibrate(p); }

// Repeating beeper whose speed / direction / pitch are updated every frame
const beeper = { timer: null, interval: 0, x: 0.5, freq: 880, type: 'sine', soft: false,
  set(interval, x, freq, type = 'sine', soft = false) {
    this.x = x; this.freq = freq; this.type = type; this.soft = soft;
    if (!interval) { this.stop(); return; }
    const changed = Math.abs(interval - this.interval) > 60;
    this.interval = interval;
    if (!this.timer || changed) this.restart();
  },
  restart() {
    clearTimeout(this.timer);
    const tick = () => {
      if (this.soft) tone({ freq: this.freq, type: this.type, vol: 0.2 });
      else cue(this.x, this.freq, this.type);
      this.timer = setTimeout(tick, this.interval);
    };
    tick();
  },
  stop() { clearTimeout(this.timer); this.timer = null; this.interval = 0; },
};
/* SOUND CHECK = automatic calibration.
   Plays a beep that the app intends for the LEFT ear and asks where you heard it.
   Some earbuds / Bluetooth drivers swap the channels, so the answer decides the mapping for good. */
function soundTest() {
  silence();
  S.swapEars = false;                               // start from the standard mapping
  const at = (ms, fn) => S.timers.push(setTimeout(fn, ms));
  say('soundCheck', [], { priority: 3 });
  at(1500, () => cue(0.1)); at(1900, () => cue(0.1)); at(2300, () => cue(0.1));
  at(2900, () => { say('whichEar', [], { priority: 3 }); askEar(); });
}
function askEar() {
  let dlg = $('earDialog');
  if (!dlg) {
    dlg = document.createElement('dialog'); dlg.id = 'earDialog';
    dlg.innerHTML = '<h2>Which ear did you hear the beeps in?</h2>' +
      '<div class="find-grid" style="grid-template-columns:1fr 1fr">' +
      '<button id="earL" style="min-height:110px;font-size:1.4rem">Left ear (L)</button>' +
      '<button id="earR" style="min-height:110px;font-size:1.4rem">Right ear (R)</button></div>' +
      '<button id="earAgain" class="ghost">Play again</button>';
    document.body.appendChild(dlg);
    $('earL').onclick = () => setEar('left');
    $('earR').onclick = () => setEar('right');
    $('earAgain').onclick = () => { dlg.close(); soundTest(); };
  }
  S.calibrating = true;
  if (!dlg.open) dlg.showModal();
}
function setEar(heard) {
  S.calibrating = false;
  const dlg = $('earDialog'); if (dlg && dlg.open) dlg.close();
  S.swapEars = heard === 'right';                   // heard it on the right → this device is reversed
  try { localStorage.setItem('drishti.swapEars', String(S.swapEars)); } catch (_) {}
  updateSwapLabel();
  silence();
  say(S.swapEars ? 'earFixed' : 'earGood', [], { priority: 3 });
  S.timers.push(setTimeout(() => cue(0.1), 3600));
  S.timers.push(setTimeout(() => say('earRight', [], { priority: 3 }), 4300));
  S.timers.push(setTimeout(() => cue(0.9), 6200));
}
function updateSwapLabel() {
  const b = $('btnSwapEars');
  if (b) b.textContent = S.swapEars ? 'Left and right: SWAPPED (tap to undo)' : 'Beeps in the wrong ear? Swap left and right';
}
/** Flip left/right sound (for earbuds or drivers that are reversed). Remembered on this device. */
function swapEars() {
  S.swapEars = !S.swapEars;
  try { localStorage.setItem('drishti.swapEars', String(S.swapEars)); } catch (_) {}
  silence();
  say(S.swapEars ? 'swapOn' : 'swapOff', [], { priority: 3 });
  S.timers.push(setTimeout(() => cue(0.1), 3200));
}
try { S.swapEars = localStorage.getItem('drishti.swapEars') === 'true'; } catch (_) {}

// ================================================================ GEOMETRY
function frameSize() { return { w: video.videoWidth || 640, h: video.videoHeight || 480 }; }
/** Horizontal position 0 (left) … 1 (right), from the USER's point of view */
function xFracOf(det) {
  const raw = (det.bbox[0] + det.bbox[2] / 2) / frameSize().w;
  return S.mirror ? 1 - raw : raw;
}
function yFracOf(det) { return (det.bbox[1] + det.bbox[3] / 2) / frameSize().h; }
/** x → clock direction (10 … 2 o'clock across the camera view) */
function clockOf(xFrac) {
  const h = Math.round((xFrac - 0.5) * 4);           // -2 … +2
  const hour = ((12 + h - 1) % 12) + 1;
  return hour === 12 ? "12 o'clock" : `${hour} o'clock`;
}
function sideWords(x) { const z = zoneOf(x); return z === 'ahead' ? 'straight ahead' : `on your ${z}`; }
/** Pinhole-camera distance estimate in metres (approximate!) */
function distanceOf(det) {
  const { h } = frameSize();
  const hFrac = Math.max(0.02, det.bbox[3] / h);
  const real = REAL_HEIGHT[det.class] || 0.5;
  const touches = det.bbox[1] < 4 || det.bbox[1] + det.bbox[3] > h - 4; // cut off → very close
  let d = real / (2 * Math.tan((CFG.vfovDeg * Math.PI / 180) / 2) * hFrac);
  if (touches && hFrac > 0.7) d = Math.min(d, 0.8);
  return d;
}
function stepsText(m) {
  if (m < 0.8) return 'very close';
  const steps = Math.max(1, Math.round(m / CFG.stepMeters));
  return steps === 1 ? 'about 1 step away' : `about ${steps} steps away`;
}
function niceName(cls) { return cls === 'cell phone' ? 'phone' : cls === 'dining table' ? 'table' : cls === 'tv' ? 'TV screen' : cls; }
function plural(cls, n) { const w = niceName(cls); return n === 1 ? `a ${w}` : `${n} ${w === 'person' ? 'people' : w + 's'}`; }
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

// ================================================================ LANGUAGES (English / हिन्दी / ಕನ್ನಡ)
/* Every spoken message goes through say(key, ...rawArgs). Each language builds its own sentence
   from the raw facts (object class, side, distance), so word order stays natural in every language.
   If the device has no voice for the chosen language, the message is SHOWN in that language
   and SPOKEN in English, so the user is never left in silence.                                   */
const LANG_CODE = { en: 'en-IN', hi: 'hi-IN', kn: 'kn-IN' };
const LANG_NAME = { en: 'English', hi: 'हिन्दी', kn: 'ಕನ್ನಡ' };
const OCR_FOR_LANG = { en: 'eng', hi: 'eng+hin', kn: 'eng+kan' };

const OBJ = {
  hi: { person: 'व्यक्ति', chair: 'कुर्सी', 'dining table': 'मेज़', couch: 'सोफ़ा', bench: 'बेंच', bed: 'बिस्तर',
    bottle: 'बोतल', cup: 'कप', 'wine glass': 'गिलास', 'cell phone': 'फ़ोन', laptop: 'लैपटॉप', tv: 'टीवी', book: 'किताब',
    backpack: 'बैग', handbag: 'हैंडबैग', suitcase: 'सूटकेस', umbrella: 'छाता', bicycle: 'साइकिल', car: 'कार',
    motorcycle: 'मोटरसाइकिल', bus: 'बस', truck: 'ट्रक', dog: 'कुत्ता', cat: 'बिल्ली', cow: 'गाय', horse: 'घोड़ा',
    'potted plant': 'गमला', toilet: 'शौचालय', sink: 'सिंक', refrigerator: 'फ्रिज', keyboard: 'कीबोर्ड', mouse: 'माउस',
    remote: 'रिमोट', clock: 'घड़ी', bowl: 'कटोरी', vase: 'फूलदान', scissors: 'कैंची', toothbrush: 'टूथब्रश',
    banana: 'केला', apple: 'सेब', orange: 'संतरा', 'traffic light': 'ट्रैफिक लाइट', 'stop sign': 'स्टॉप साइन' },
  kn: { person: 'ವ್ಯಕ್ತಿ', chair: 'ಕುರ್ಚಿ', 'dining table': 'ಮೇಜು', couch: 'ಸೋಫಾ', bench: 'ಬೆಂಚು', bed: 'ಹಾಸಿಗೆ',
    bottle: 'ಬಾಟಲ್', cup: 'ಕಪ್', 'wine glass': 'ಲೋಟ', 'cell phone': 'ಫೋನ್', laptop: 'ಲ್ಯಾಪ್‌ಟಾಪ್', tv: 'ಟಿವಿ', book: 'ಪುಸ್ತಕ',
    backpack: 'ಬ್ಯಾಗ್', handbag: 'ಕೈಚೀಲ', suitcase: 'ಸೂಟ್‌ಕೇಸ್', umbrella: 'ಛತ್ರಿ', bicycle: 'ಸೈಕಲ್', car: 'ಕಾರು',
    motorcycle: 'ಬೈಕ್', bus: 'ಬಸ್', truck: 'ಲಾರಿ', dog: 'ನಾಯಿ', cat: 'ಬೆಕ್ಕು', cow: 'ಹಸು', horse: 'ಕುದುರೆ',
    'potted plant': 'ಹೂಕುಂಡ', toilet: 'ಶೌಚಾಲಯ', sink: 'ಸಿಂಕ್', refrigerator: 'ಫ್ರಿಡ್ಜ್', keyboard: 'ಕೀಬೋರ್ಡ್', mouse: 'ಮೌಸ್',
    remote: 'ರಿಮೋಟ್', clock: 'ಗಡಿಯಾರ', bowl: 'ಬಟ್ಟಲು', vase: 'ಹೂದಾನಿ', scissors: 'ಕತ್ತರಿ', toothbrush: 'ಟೂತ್‌ಬ್ರಷ್',
    banana: 'ಬಾಳೆಹಣ್ಣು', apple: 'ಸೇಬು', orange: 'ಕಿತ್ತಳೆ', 'traffic light': 'ಟ್ರಾಫಿಕ್ ಲೈಟ್', 'stop sign': 'ಸ್ಟಾಪ್ ಫಲಕ' },
};
const SIGN_LOCAL = {
  hi: { washroom_men: 'पुरुष शौचालय का संकेत', washroom_women: 'महिला शौचालय का संकेत', washroom: 'शौचालय का संकेत',
    exit_sign: 'निकास का संकेत', pedestrian_crossing: 'पैदल पार पथ का संकेत। आगे ज़ेब्रा क्रॉसिंग है',
    road_work: 'सड़क पर काम चल रहा है, सावधान रहिए', school_ahead: 'आगे स्कूल है, बच्चों का ध्यान रखिए',
    speed_breaker: 'आगे स्पीड ब्रेकर है', caution: 'सावधान, आगे खतरा है, धीरे चलिए',
    no_entry: 'प्रवेश निषेध, इस तरफ़ मत जाइए', stop: 'स्टॉप साइन' },
  kn: { washroom_men: 'ಪುರುಷರ ಶೌಚಾಲಯ ಫಲಕ', washroom_women: 'ಮಹಿಳೆಯರ ಶೌಚಾಲಯ ಫಲಕ', washroom: 'ಶೌಚಾಲಯ ಫಲಕ',
    exit_sign: 'ನಿರ್ಗಮನ ಫಲಕ', pedestrian_crossing: 'ಪಾದಚಾರಿ ದಾಟುವ ಫಲಕ. ಮುಂದೆ ಜೀಬ್ರಾ ಕ್ರಾಸಿಂಗ್ ಇದೆ',
    road_work: 'ರಸ್ತೆ ಕಾಮಗಾರಿ ನಡೆಯುತ್ತಿದೆ, ಎಚ್ಚರಿಕೆ', school_ahead: 'ಮುಂದೆ ಶಾಲೆ ಇದೆ, ಮಕ್ಕಳ ಬಗ್ಗೆ ಎಚ್ಚರ',
    speed_breaker: 'ಮುಂದೆ ಸ್ಪೀಡ್ ಬ್ರೇಕರ್ ಇದೆ', caution: 'ಎಚ್ಚರಿಕೆ, ಮುಂದೆ ಅಪಾಯ ಇದೆ, ನಿಧಾನವಾಗಿ ಹೋಗಿ',
    no_entry: 'ಪ್ರವೇಶ ನಿಷೇಧ, ಈ ಕಡೆ ಹೋಗಬೇಡಿ', stop: 'ಸ್ಟಾಪ್ ಫಲಕ' },
};
const SIGNSEARCH = { hi: { washroom: 'शौचालय', exit: 'निकास', road: 'सड़क' }, kn: { washroom: 'ಶೌಚಾಲಯ', exit: 'ನಿರ್ಗಮನ', road: 'ರಸ್ತೆ' } };

// small helpers per language: object name, side, distance, plural, find directions
const H = {
  en: {
    obj: c => niceName(c), Obj: c => cap(niceName(c)),
    side: z => (z === 'ahead' ? 'ahead' : `on the ${z}`),
    from: z => (z === 'ahead' ? 'ahead' : `from your ${z}`),
    steps: m => (m < 0.8 ? 'very close' : (s => (s === 1 ? '1 step' : `${s} steps`))(Math.max(1, Math.round(m / CFG.stepMeters)))),
    count: (c, n) => cap(plural(c, n)),
    horiz: g => (g.kind === 'ahead' ? 'straight ahead' : g.kind === 'slight' ? `slightly to your ${g.side}` : `to your ${g.side}`),
    vert: v => ({ up: 'up high', down: 'low down' }[v] || ''),
    secs: n => `${n} second${n > 1 ? 's' : ''}`,
  },
  hi: {
    obj: c => OBJ.hi[c] || niceName(c), Obj: c => OBJ.hi[c] || niceName(c),
    side: z => ({ left: 'बाईं ओर', right: 'दाईं ओर', ahead: 'सामने' }[z]),
    from: z => ({ left: 'बाईं ओर से', right: 'दाईं ओर से', ahead: 'सामने से' }[z]),
    steps: m => (m < 0.8 ? 'बहुत पास' : `${Math.max(1, Math.round(m / CFG.stepMeters))} कदम`),
    count: (c, n) => (n === 1 ? (OBJ.hi[c] || niceName(c)) : c === 'person' ? `${n} लोग` : `${n} ${OBJ.hi[c] || niceName(c)}`),
    horiz: g => (g.kind === 'ahead' ? 'सीधे सामने' : g.kind === 'slight' ? `थोड़ा ${g.side === 'left' ? 'बाईं' : 'दाईं'} ओर` : `${g.side === 'left' ? 'बाईं' : 'दाईं'} ओर`),
    vert: v => ({ up: 'ऊपर', down: 'नीचे' }[v] || ''),
    secs: n => `${n} सेकंड`,
  },
  kn: {
    obj: c => OBJ.kn[c] || niceName(c), Obj: c => OBJ.kn[c] || niceName(c),
    side: z => ({ left: 'ಎಡಕ್ಕೆ', right: 'ಬಲಕ್ಕೆ', ahead: 'ಮುಂದೆ' }[z]),
    from: z => ({ left: 'ಎಡಗಡೆಯಿಂದ', right: 'ಬಲಗಡೆಯಿಂದ', ahead: 'ಮುಂದಿನಿಂದ' }[z]),
    steps: m => (m < 0.8 ? 'ತುಂಬಾ ಹತ್ತಿರ' : `${Math.max(1, Math.round(m / CFG.stepMeters))} ಹೆಜ್ಜೆ`),
    count: (c, n) => (n === 1 ? (OBJ.kn[c] || niceName(c)) : c === 'person' ? `${n} ಜನ` : `${n} ${OBJ.kn[c] || niceName(c)}`),
    horiz: g => (g.kind === 'ahead' ? 'ನೇರವಾಗಿ ಮುಂದೆ' : g.kind === 'slight' ? `ಸ್ವಲ್ಪ ${g.side === 'left' ? 'ಎಡಕ್ಕೆ' : 'ಬಲಕ್ಕೆ'}` : (g.side === 'left' ? 'ಎಡಕ್ಕೆ' : 'ಬಲಕ್ಕೆ')),
    vert: v => ({ up: 'ಮೇಲೆ', down: 'ಕೆಳಗೆ' }[v] || ''),
    secs: n => `${n} ಸೆಕೆಂಡ್`,
  },
};
const dirWord = { en: { left: 'left', right: 'right', 'straight ahead': 'straight ahead' },
  hi: { left: 'बाईं ओर', right: 'दाईं ओर', 'straight ahead': 'सीधे' }, kn: { left: 'ಎಡಕ್ಕೆ', right: 'ಬಲಕ್ಕೆ', 'straight ahead': 'ನೇರವಾಗಿ' } };

const MSG = {
  en: {
    ready: () => 'Drishti ready.', guardOn: () => 'Guard on.', loading: () => 'Still loading, one moment.',
    soundCheck: () => 'Sound check. Listen.', whichEar: () => 'Which ear did you hear that in? Press L for left, or R for right.',
    earFixed: () => 'Fixed. Your earphones were reversed. This beep is now in your left ear.',
    earGood: () => 'Good, your sound is correct. This beep is in your left ear.', earRight: () => 'And this one in your right ear.',
    swapOn: () => 'Left and right swapped. This beep should now be in your left ear.',
    swapOff: () => 'Left and right back to normal. This beep should be in your left ear.',
    visionFail: () => 'Could not load the vision model. Connect to the internet for the first load, then reload.',
    approaching: (h, c, z, s) => `${h.Obj(c)} approaching ${h.from(z)}. ${h.secs(s)}.`,
    staticNear: (h, c, z, s) => `${h.Obj(c)} ${h.side(z)}, ${h.secs(s)} away.`,
    crossing: (h, c, toRight) => `${h.Obj(c)} crossing from ${toRight ? 'left to right' : 'right to left'}.`,
    trafficLight: () => 'Traffic light ahead. Check the signal before crossing.', stopSignAhead: () => 'Stop sign ahead.',
    stopObj: (h, c) => `Stop. ${h.Obj(c)} ahead.`, objAt: (h, c, z, m) => `${h.Obj(c)} ${h.side(z)}, ${h.steps(m)}.`,
    lostIt: (h, g) => `Lost it. It was ${h.horiz(g)}.`, stillLooking: () => 'Still looking. Turn slowly.',
    found: (h, c, g, v, m) => `Found the ${h.obj(c)}, ${h.horiz(g)}${v ? ', ' + h.vert(v) : ''}, ${h.steps(m)}.`,
    reach: (h, v) => `Within reach. Reach out ${h.vert(v) || 'straight ahead'}.`, straight: (h, m) => `Straight ahead, ${h.steps(m)}.`,
    turn: (h, side) => `Turn ${side}.`, finding: (h, c) => `Finding ${h.obj(c)}. Sweep slowly.`,
    cantTrack: (h, w) => `Sorry, I can't track ${w} yet. Try bottle, cup, phone, laptop, chair, bag, book or person.`,
    sign: (h, label, dir) => `${signName(label)}${dir ? ', pointing ' + dir : ''}.`,
    noText: () => 'No text found. Move closer.', readerFail: () => 'Text reader not loaded. Check the internet.',
    nothing: () => 'Nothing recognised. Turn slowly.',
    describe: (h, items) => items.map(it => `${h.count(it.cls, it.n)} ${h.side(zoneOf(it.x))}, ${h.steps(it.m)}`).join('. ') + '.',
    lookingFor: (h, w) => `Looking for ${w} signs. Turn slowly.`, noneFound: (h, w) => `No ${w} sign found. Keep turning slowly.`,
    noneHere: (h, w) => `No ${w} sign here. Try another corridor, or ask someone nearby.`,
    whatFind: () => 'What should I find?', tryCmds: () => 'Try: describe, read, find bottle, washroom, guard, or stop.',
    help: () => 'Say guard to detect obstacles. Describe, to hear what is around you. Read, to read text. Find, followed by an object, like find bottle. Washroom, exit, or road signs. Say stop to stop. Tap the screen to describe, double tap to read, press and hold to talk.',
    voiceNone: () => 'Voice commands need Chrome or Edge. Use the buttons instead.', voiceFail: (h, e) => `Voice input failed: ${e}. Use the buttons instead.`,
    camFail: () => 'The camera did not open. Allow camera access, and open the app on localhost or an https address.',
    langSet: () => 'English selected.',
  },
  hi: {
    ready: () => 'दृष्टि तैयार है।', guardOn: () => 'गार्ड चालू।', loading: () => 'अभी लोड हो रहा है, एक पल रुकिए।',
    soundCheck: () => 'आवाज़ की जाँच। ध्यान से सुनिए।', whichEar: () => 'यह आवाज़ किस कान में सुनाई दी? बाएं के लिए L, दाएं के लिए R दबाइए।',
    earFixed: () => 'ठीक कर दिया। आपके ईयरफ़ोन उल्टे थे। यह आवाज़ अब बाएं कान में है।',
    earGood: () => 'बढ़िया, आवाज़ सही है। यह आवाज़ बाएं कान में है।', earRight: () => 'और यह दाएं कान में।',
    swapOn: () => 'बायां और दायां बदल दिया। यह आवाज़ अब बाएं कान में होनी चाहिए।',
    swapOff: () => 'बायां और दायां सामान्य। यह आवाज़ बाएं कान में होनी चाहिए।',
    visionFail: () => 'विज़न मॉडल लोड नहीं हुआ। पहली बार इंटरनेट से जुड़िए और फिर से खोलिए।',
    approaching: (h, c, z, s) => `${h.Obj(c)} ${h.from(z)} पास आ रहा है। ${h.secs(s)}।`,
    staticNear: (h, c, z, s) => `${h.Obj(c)} ${h.side(z)}, ${h.secs(s)} दूर।`,
    crossing: (h, c, toRight) => `${h.Obj(c)} ${toRight ? 'बाएं से दाएं' : 'दाएं से बाएं'} जा रहा है।`,
    trafficLight: () => 'सामने ट्रैफिक लाइट है। पार करने से पहले सिग्नल देखिए।', stopSignAhead: () => 'सामने स्टॉप साइन है।',
    stopObj: (h, c) => `रुकिए। सामने ${h.obj(c)} है।`, objAt: (h, c, z, m) => `${h.Obj(c)} ${h.side(z)}, ${h.steps(m)}।`,
    lostIt: (h, g) => `खो गया। वह ${h.horiz(g)} था।`, stillLooking: () => 'अभी ढूंढ रहे हैं। धीरे से घूमिए।',
    found: (h, c, g, v, m) => `${h.Obj(c)} मिल गया, ${h.horiz(g)}${v ? ', ' + h.vert(v) : ''}, ${h.steps(m)}।`,
    reach: (h, v) => `हाथ की पहुँच में है। ${h.vert(v) || 'सीधे'} हाथ बढ़ाइए।`, straight: (h, m) => `सीधे सामने, ${h.steps(m)}।`,
    turn: (h, side) => (side === 'left' ? 'बाएं मुड़िए।' : 'दाएं मुड़िए।'), finding: (h, c) => `${h.obj(c)} ढूंढ रहे हैं। फ़ोन धीरे घुमाइए।`,
    cantTrack: (h, w) => `माफ़ कीजिए, ${w} अभी नहीं ढूंढ सकते। बोतल, कप, फ़ोन, लैपटॉप, कुर्सी, बैग या किताब बोलिए।`,
    sign: (h, label, dir) => `${SIGN_LOCAL.hi[label] || signName(label)}${dir ? ', ' + dirWord.hi[dir] + ' इशारा' : ''}।`,
    noText: () => 'कोई लिखावट नहीं मिली। पास ले जाइए।', readerFail: () => 'टेक्स्ट रीडर लोड नहीं हुआ। इंटरनेट जाँचिए।',
    nothing: () => 'कुछ पहचान में नहीं आया। धीरे से घूमिए।',
    describe: (h, items) => items.map(it => `${h.count(it.cls, it.n)} ${h.side(zoneOf(it.x))}, ${h.steps(it.m)}`).join('। ') + '।',
    lookingFor: (h, w) => `${SIGNSEARCH.hi[w]} के संकेत ढूंढ रहे हैं। धीरे से घूमिए।`,
    noneFound: (h, w) => `${SIGNSEARCH.hi[w]} का कोई संकेत नहीं मिला। धीरे-धीरे घूमते रहिए।`,
    noneHere: (h, w) => `यहाँ ${SIGNSEARCH.hi[w]} का कोई संकेत नहीं है। दूसरी तरफ़ जाइए या किसी से पूछिए।`,
    whatFind: () => 'क्या ढूंढूं?', tryCmds: () => 'ऐसे बोलिए: क्या है, पढ़ो, बोतल ढूंढो, शौचालय, गार्ड, या रुको।',
    help: () => 'बाधाओं के लिए गार्ड बोलिए। आसपास जानने के लिए क्या है। लिखावट के लिए पढ़ो। चीज़ के लिए, बोतल ढूंढो। शौचालय, निकास, या सड़क के संकेत। रोकने के लिए रुको। स्क्रीन पर एक बार टैप, दो बार टैप पढ़ने के लिए, दबाकर रखिए बोलने के लिए।',
    voiceNone: () => 'आवाज़ के आदेश के लिए क्रोम या एज चाहिए। बटन इस्तेमाल कीजिए।', voiceFail: (h, e) => `आवाज़ नहीं पहचानी गई: ${e}। बटन इस्तेमाल कीजिए।`,
    camFail: () => 'कैमरा नहीं खुला। कैमरे की अनुमति दीजिए।',
    langSet: () => 'हिन्दी चुनी गई।',
  },
  kn: {
    ready: () => 'ದೃಷ್ಟಿ ಸಿದ್ಧವಾಗಿದೆ.', guardOn: () => 'ಗಾರ್ಡ್ ಆನ್.', loading: () => 'ಇನ್ನೂ ಲೋಡ್ ಆಗುತ್ತಿದೆ, ಒಂದು ಕ್ಷಣ.',
    soundCheck: () => 'ಧ್ವನಿ ಪರೀಕ್ಷೆ. ಗಮನವಿಟ್ಟು ಕೇಳಿ.', whichEar: () => 'ಆ ಶಬ್ದ ಯಾವ ಕಿವಿಯಲ್ಲಿ ಕೇಳಿಸಿತು? ಎಡಕ್ಕೆ L, ಬಲಕ್ಕೆ R ಒತ್ತಿ.',
    earFixed: () => 'ಸರಿಪಡಿಸಲಾಗಿದೆ. ನಿಮ್ಮ ಇಯರ್‌ಫೋನ್ ಉಲ್ಟಾ ಇತ್ತು. ಈ ಶಬ್ದ ಈಗ ಎಡ ಕಿವಿಯಲ್ಲಿದೆ.',
    earGood: () => 'ಚೆನ್ನಾಗಿದೆ, ಧ್ವನಿ ಸರಿಯಾಗಿದೆ. ಈ ಶಬ್ದ ಎಡ ಕಿವಿಯಲ್ಲಿದೆ.', earRight: () => 'ಮತ್ತು ಇದು ಬಲ ಕಿವಿಯಲ್ಲಿ.',
    swapOn: () => 'ಎಡ ಮತ್ತು ಬಲ ಬದಲಾಯಿಸಲಾಗಿದೆ. ಈ ಶಬ್ದ ಈಗ ಎಡ ಕಿವಿಯಲ್ಲಿ ಇರಬೇಕು.',
    swapOff: () => 'ಎಡ ಮತ್ತು ಬಲ ಸಾಮಾನ್ಯ. ಈ ಶಬ್ದ ಎಡ ಕಿವಿಯಲ್ಲಿ ಇರಬೇಕು.',
    visionFail: () => 'ವಿಷನ್ ಮಾಡೆಲ್ ಲೋಡ್ ಆಗಲಿಲ್ಲ. ಮೊದಲ ಬಾರಿ ಇಂಟರ್ನೆಟ್ ಸಂಪರ್ಕಿಸಿ, ಮತ್ತೆ ತೆರೆಯಿರಿ.',
    approaching: (h, c, z, s) => `${h.Obj(c)} ${h.from(z)} ಹತ್ತಿರ ಬರುತ್ತಿದೆ. ${h.secs(s)}.`,
    staticNear: (h, c, z, s) => `${h.Obj(c)} ${h.side(z)}, ${h.secs(s)} ದೂರ.`,
    crossing: (h, c, toRight) => `${h.Obj(c)} ${toRight ? 'ಎಡದಿಂದ ಬಲಕ್ಕೆ' : 'ಬಲದಿಂದ ಎಡಕ್ಕೆ'} ದಾಟುತ್ತಿದೆ.`,
    trafficLight: () => 'ಮುಂದೆ ಟ್ರಾಫಿಕ್ ಲೈಟ್ ಇದೆ. ದಾಟುವ ಮೊದಲು ಸಿಗ್ನಲ್ ನೋಡಿ.', stopSignAhead: () => 'ಮುಂದೆ ಸ್ಟಾಪ್ ಫಲಕ ಇದೆ.',
    stopObj: (h, c) => `ನಿಲ್ಲಿ. ಮುಂದೆ ${h.obj(c)} ಇದೆ.`, objAt: (h, c, z, m) => `${h.Obj(c)} ${h.side(z)}, ${h.steps(m)}.`,
    lostIt: (h, g) => `ಕಾಣುತ್ತಿಲ್ಲ. ಅದು ${h.horiz(g)} ಇತ್ತು.`, stillLooking: () => 'ಇನ್ನೂ ಹುಡುಕುತ್ತಿದ್ದೇನೆ. ನಿಧಾನವಾಗಿ ತಿರುಗಿ.',
    found: (h, c, g, v, m) => `${h.Obj(c)} ಸಿಕ್ಕಿತು, ${h.horiz(g)}${v ? ', ' + h.vert(v) : ''}, ${h.steps(m)}.`,
    reach: (h, v) => `ಕೈಗೆ ಸಿಗುವಷ್ಟು ಹತ್ತಿರ. ${h.vert(v) || 'ನೇರವಾಗಿ'} ಕೈ ಚಾಚಿ.`, straight: (h, m) => `ನೇರವಾಗಿ ಮುಂದೆ, ${h.steps(m)}.`,
    turn: (h, side) => (side === 'left' ? 'ಎಡಕ್ಕೆ ತಿರುಗಿ.' : 'ಬಲಕ್ಕೆ ತಿರುಗಿ.'), finding: (h, c) => `${h.obj(c)} ಹುಡುಕುತ್ತಿದ್ದೇನೆ. ನಿಧಾನವಾಗಿ ತಿರುಗಿಸಿ.`,
    cantTrack: (h, w) => `ಕ್ಷಮಿಸಿ, ${w} ಇನ್ನೂ ಹುಡುಕಲು ಆಗುವುದಿಲ್ಲ. ಬಾಟಲ್, ಕಪ್, ಫೋನ್, ಲ್ಯಾಪ್‌ಟಾಪ್, ಕುರ್ಚಿ, ಬ್ಯಾಗ್ ಅಥವಾ ಪುಸ್ತಕ ಹೇಳಿ.`,
    sign: (h, label, dir) => `${SIGN_LOCAL.kn[label] || signName(label)}${dir ? ', ' + dirWord.kn[dir] + ' ಸೂಚಿಸುತ್ತಿದೆ' : ''}.`,
    noText: () => 'ಯಾವುದೇ ಬರಹ ಸಿಗಲಿಲ್ಲ. ಹತ್ತಿರ ತನ್ನಿ.', readerFail: () => 'ಟೆಕ್ಸ್ಟ್ ರೀಡರ್ ಲೋಡ್ ಆಗಲಿಲ್ಲ. ಇಂಟರ್ನೆಟ್ ಪರಿಶೀಲಿಸಿ.',
    nothing: () => 'ಏನೂ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ. ನಿಧಾನವಾಗಿ ತಿರುಗಿ.',
    describe: (h, items) => items.map(it => `${h.count(it.cls, it.n)} ${h.side(zoneOf(it.x))}, ${h.steps(it.m)}`).join('. ') + '.',
    lookingFor: (h, w) => `${SIGNSEARCH.kn[w]} ಫಲಕ ಹುಡುಕುತ್ತಿದ್ದೇನೆ. ನಿಧಾನವಾಗಿ ತಿರುಗಿ.`,
    noneFound: (h, w) => `${SIGNSEARCH.kn[w]} ಫಲಕ ಸಿಗಲಿಲ್ಲ. ನಿಧಾನವಾಗಿ ತಿರುಗುತ್ತಿರಿ.`,
    noneHere: (h, w) => `ಇಲ್ಲಿ ${SIGNSEARCH.kn[w]} ಫಲಕ ಇಲ್ಲ. ಬೇರೆ ಕಡೆ ಹೋಗಿ ಅಥವಾ ಯಾರನ್ನಾದರೂ ಕೇಳಿ.`,
    whatFind: () => 'ಏನನ್ನು ಹುಡುಕಲಿ?', tryCmds: () => 'ಹೀಗೆ ಹೇಳಿ: ಏನಿದೆ, ಓದು, ಬಾಟಲ್ ಹುಡುಕು, ಶೌಚಾಲಯ, ಗಾರ್ಡ್, ಅಥವಾ ನಿಲ್ಲಿಸು.',
    help: () => 'ಅಡೆತಡೆಗಳಿಗೆ ಗಾರ್ಡ್ ಎನ್ನಿ. ಸುತ್ತ ಏನಿದೆ ತಿಳಿಯಲು ಏನಿದೆ. ಬರಹಕ್ಕೆ ಓದು. ವಸ್ತುವಿಗೆ, ಬಾಟಲ್ ಹುಡುಕು. ಶೌಚಾಲಯ, ನಿರ್ಗಮನ ಅಥವಾ ರಸ್ತೆ ಫಲಕ. ನಿಲ್ಲಿಸಲು ನಿಲ್ಲಿಸು. ಪರದೆಯ ಮೇಲೆ ಒಂದು ಬಾರಿ ಟ್ಯಾಪ್, ಓದಲು ಎರಡು ಬಾರಿ, ಮಾತನಾಡಲು ಒತ್ತಿ ಹಿಡಿಯಿರಿ.',
    voiceNone: () => 'ಧ್ವನಿ ಆದೇಶಕ್ಕೆ ಕ್ರೋಮ್ ಅಥವಾ ಎಡ್ಜ್ ಬೇಕು. ಬಟನ್ ಬಳಸಿ.', voiceFail: (h, e) => `ಧ್ವನಿ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ: ${e}. ಬಟನ್ ಬಳಸಿ.`,
    camFail: () => 'ಕ್ಯಾಮೆರಾ ತೆರೆಯಲಿಲ್ಲ. ಕ್ಯಾಮೆರಾ ಅನುಮತಿ ನೀಡಿ.',
    langSet: () => 'ಕನ್ನಡ ಆಯ್ಕೆಯಾಗಿದೆ.',
  },
};
function msg(lang, key, ...a) { const f = (MSG[lang] && MSG[lang][key]) || MSG.en[key]; return f(H[lang] || H.en, ...a); }
function hasVoice(lang) {
  if (!voices.length) loadVoices();
  const code = LANG_CODE[lang].slice(0, 2);
  return voices.some(v => v.lang.replace('_', '-').toLowerCase().startsWith(code));
}
/** Speak a message in the chosen language (falls back to spoken English + on-screen local text). */
function say(key, args = [], opts = {}) {
  const local = msg(S.lang, key, ...args);
  if (S.lang === 'en' || hasVoice(S.lang)) return speak(local, { ...opts, lang: LANG_CODE[S.lang] });
  speak(msg('en', key, ...args), { ...opts, lang: 'en-IN' });
  $('caption').textContent = local;                // show it in the chosen language (no duplicate in history)
}

// UI labels for the main controls
const UI = {
  en: { btnGuard: ['Guard', 'Beeps and warns about obstacles'], btnDescribe: ['Describe', 'Says what is around you'],
    btnRead: ['Read', 'Reads signs, labels and boards'], btnFind: ['Find', 'Guides you to your things'],
    btnWashroom: ['Washroom', 'Looks for washroom signs'], btnMic: ['Speak', 'Say a command'], stop: 'Stop', settings: 'Settings' },
  hi: { btnGuard: ['गार्ड', 'बाधाओं की चेतावनी देता है'], btnDescribe: ['क्या है', 'आसपास क्या है बताता है'],
    btnRead: ['पढ़ो', 'संकेत और लिखावट पढ़ता है'], btnFind: ['ढूंढो', 'आपकी चीज़ें ढूंढता है'],
    btnWashroom: ['शौचालय', 'शौचालय के संकेत ढूंढता है'], btnMic: ['बोलिए', 'आदेश बोलिए'], stop: 'रुको', settings: 'सेटिंग्स' },
  kn: { btnGuard: ['ಗಾರ್ಡ್', 'ಅಡೆತಡೆಗಳ ಬಗ್ಗೆ ಎಚ್ಚರಿಸುತ್ತದೆ'], btnDescribe: ['ಏನಿದೆ', 'ಸುತ್ತ ಏನಿದೆ ಎಂದು ಹೇಳುತ್ತದೆ'],
    btnRead: ['ಓದು', 'ಫಲಕ ಮತ್ತು ಬರಹ ಓದುತ್ತದೆ'], btnFind: ['ಹುಡುಕು', 'ನಿಮ್ಮ ವಸ್ತುಗಳನ್ನು ಹುಡುಕುತ್ತದೆ'],
    btnWashroom: ['ಶೌಚಾಲಯ', 'ಶೌಚಾಲಯ ಫಲಕ ಹುಡುಕುತ್ತದೆ'], btnMic: ['ಮಾತನಾಡಿ', 'ಆದೇಶ ಹೇಳಿ'], stop: 'ನಿಲ್ಲಿಸು', settings: 'ಸೆಟ್ಟಿಂಗ್ಸ್' },
};
function applyLanguage(lang, announce = false) {
  S.lang = MSG[lang] ? lang : 'en';
  try { localStorage.setItem('drishti.lang', S.lang); } catch (_) {}
  document.documentElement.lang = S.lang;
  const u = UI[S.lang];
  for (const id of ['btnGuard', 'btnDescribe', 'btnRead', 'btnFind', 'btnWashroom', 'btnMic']) {
    const b = $(id); if (!b) continue;
    b.querySelector('.mode-name').textContent = u[id][0];
    b.querySelector('.mode-hint').textContent = u[id][1];
  }
  const st = $('btnStop'); if (st) st.firstChild.textContent = u.stop + ' ';
  const se = $('btnSettings'); if (se) se.textContent = u.settings;
  const ocr = $('setOcrLang'); if (ocr) { ocr.value = OCR_FOR_LANG[S.lang]; ocr.dispatchEvent(new Event('change')); }
  const sel = $('setLang'); if (sel) sel.value = S.lang;
  document.querySelectorAll('.lang-pick button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.lang === S.lang)));
  if (announce) say('langSet', [], { priority: 3 });
  if (S.lang !== 'en' && !hasVoice(S.lang) && announce) {
    setCaption(`${LANG_NAME[S.lang]}: no ${LANG_NAME[S.lang]} voice on this device, so messages are shown in ${LANG_NAME[S.lang]} and spoken in English. (Edge browser and Android phones have this voice.)`);
  }
}
/** Language buttons on the start screen + a selector in Settings (added from code: no HTML change needed) */
function setupLanguageUI() {
  try { S.lang = localStorage.getItem('drishti.lang') || 'en'; } catch (_) { S.lang = 'en'; }
  const pick = document.createElement('div');
  pick.className = 'lang-pick'; pick.setAttribute('role', 'group'); pick.setAttribute('aria-label', 'Language');
  pick.style.cssText = 'display:flex;gap:10px;margin:0 0 1.2em;flex-wrap:wrap';
  for (const l of ['en', 'hi', 'kn']) {
    const b = document.createElement('button');
    b.dataset.lang = l; b.textContent = LANG_NAME[l];
    b.style.cssText = 'flex:1;min-width:110px;padding:.8em;font-size:1.25rem;font-weight:700;border-radius:12px;border:2px solid var(--amber);background:transparent;color:var(--paper)';
    b.onclick = () => applyLanguage(l);           // the click then bubbles up and starts the app
    pick.appendChild(b);
  }
  $('startBtn').before(pick);
  const style = document.createElement('style');
  style.textContent = '.lang-pick button[aria-pressed="true"]{background:var(--amber)!important;color:var(--ink)!important}';
  document.head.appendChild(style);
  // settings selector
  const lab = document.createElement('label');
  lab.innerHTML = 'Language / भाषा / ಭಾಷೆ <select id="setLang"><option value="en">English</option><option value="hi">हिन्दी</option><option value="kn">ಕನ್ನಡ</option></select>';
  $('settings').querySelector('.settings-head').after(lab);
  $('setLang').addEventListener('change', e => applyLanguage(e.target.value, true));
  applyLanguage(S.lang);
}

// Voice command words in Hindi and Kannada (Devanagari / Kannada script; matched as plain substrings)
const LOCAL_INTENTS = [
  ['stop', ['रुको', 'रुकिए', 'बंद करो', 'बंद', 'चुप', 'ನಿಲ್ಲಿಸು', 'ನಿಲ್ಲಿಸಿ', 'ಸಾಕು', 'ಸುಮ್ಮನಿರು']],
  ['traffic', ['सड़क', 'ट्रैफिक', 'क्रॉसिंग', 'सिग्नल', 'ರಸ್ತೆ', 'ಟ್ರಾಫಿಕ್', 'ಕ್ರಾಸಿಂಗ್', 'ಸಿಗ್ನಲ್', 'ದಾಟ']],
  ['washroom', ['शौचालय', 'टॉयलेट', 'वॉशरूम', 'बाथरूम', 'ಶೌಚಾಲಯ', 'ಟಾಯ್ಲೆಟ್', 'ವಾಶ್‌ರೂಮ್', 'ವಾಶ್ ರೂಮ್', 'ಬಾತ್‌ರೂಮ್']],
  ['exit', ['निकास', 'बाहर', 'एग्ज़िट', 'एग्जिट', 'ನಿರ್ಗಮನ', 'ಹೊರಗೆ', 'ಎಕ್ಸಿಟ್']],
  ['find', ['ढूंढो', 'ढूँढो', 'ढूंढ', 'कहाँ है', 'कहां है', 'कहाँ', 'ಹುಡುಕು', 'ಹುಡುಕಿ', 'ಎಲ್ಲಿದೆ', 'ಎಲ್ಲಿ']],
  ['read', ['पढ़ो', 'पढ़िए', 'पढ़', 'लिखा', 'ಓದು', 'ಓದಿ', 'ಬರೆದ']],
  ['who', ['कौन', 'ಯಾರು']],
  ['describe', ['क्या है', 'आसपास', 'सामने', 'बताओ', 'देखो', 'ಏನಿದೆ', 'ಏನು', 'ಸುತ್ತ', 'ಮುಂದೆ', 'ಹೇಳು', 'ನೋಡು']],
  ['guard', ['गार्ड', 'चलो', 'रास्ता', 'ಗಾರ್ಡ್', 'ನಡೆ', 'ದಾರಿ']],
  ['help', ['मदद', 'सहायता', 'ಸಹಾಯ']],
];
const LOCAL_OBJECTS = {
  'बोतल': 'bottle', 'पानी': 'bottle', 'ಬಾಟಲ್': 'bottle', 'ಬಾಟಲಿ': 'bottle', 'ನೀರು': 'bottle',
  'फ़ोन': 'phone', 'फोन': 'phone', 'मोबाइल': 'phone', 'ಫೋನ್': 'phone', 'ಮೊಬೈಲ್': 'phone',
  'कुर्सी': 'chair', 'ಕುರ್ಚಿ': 'chair', 'बैग': 'bag', 'थैला': 'bag', 'ಬ್ಯಾಗ್': 'bag', 'ಚೀಲ': 'bag',
  'कप': 'cup', 'ಕಪ್': 'cup', 'किताब': 'book', 'ಪುಸ್ತಕ': 'book', 'लैपटॉप': 'laptop', 'ಲ್ಯಾಪ್‌ಟಾಪ್': 'laptop', 'ಲ್ಯಾಪ್ಟಾಪ್': 'laptop',
  'मेज़': 'table', 'मेज': 'table', 'ಮೇಜು': 'table', 'छाता': 'umbrella', 'ಛತ್ರಿ': 'umbrella',
};
/** Turn a Hindi/Kannada command into the English intent the rest of the app understands. */
function localCommand(text) {
  const obj = Object.keys(LOCAL_OBJECTS).find(w => text.includes(w));
  for (const [intent, words] of LOCAL_INTENTS) {
    if (!words.some(w => text.includes(w))) continue;
    if (intent === 'find' || (obj && intent !== 'stop')) return obj ? `find ${LOCAL_OBJECTS[obj]}` : 'find';
    return { stop: 'stop', traffic: 'road signs', washroom: 'washroom', exit: 'exit', read: 'read',
      who: 'who', describe: 'describe', guard: 'guard', help: 'help' }[intent];
  }
  return obj ? `find ${LOCAL_OBJECTS[obj]}` : null;
}

// ================================================================ CAMERA + MODELS
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  video.srcObject = stream;
  await new Promise(r => (video.readyState >= 2 ? r() : (video.onloadeddata = r)));
  await video.play();
  // Front cameras (laptop webcams, selfie cams) see the world mirrored → flip left/right
  const fm = stream.getVideoTracks()[0].getSettings().facingMode;
  const mobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
  let saved = null;
  try { saved = localStorage.getItem('drishti.mirror'); } catch (_) {}
  setMirror(saved !== null ? saved === 'true' : (fm ? fm === 'user' : !mobile));
}
function setMirror(on) {
  S.mirror = on; $('setMirror').checked = on;
  $('stage').classList.toggle('mirror', on);
}

async function loadModels() {
  const pill = $('statusModels');
  try {
    await tf.ready();
    S.coco = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
    pill.textContent = 'Vision on this device'; pill.className = 'pill ok';
  } catch (e) {
    console.error(e); pill.textContent = 'Vision model failed'; pill.className = 'pill warn';
    say('visionFail', [], { priority: 3 });
  }
  await loadSignModel();
  $('loading').classList.add('hide');
}

/** Our custom Teachable Machine model (trained on synthetic + open data). Optional. */
async function loadSignModel() {
  const pill = $('statusSign');
  try {
    const meta = await (await fetch('model/metadata.json', { cache: 'no-cache' })).json();
    S.signLabels = meta.labels || [];
    S.signModel = await tf.loadLayersModel('model/model.json');
    pill.textContent = `Sign model: ${S.signLabels.length} classes`; pill.className = 'pill ok';
  } catch (e) {
    S.signModel = null;
    pill.textContent = 'Sign model not trained: reading sign text instead'; pill.className = 'pill warn';
    console.info('Custom sign model not found in app/model/. Using OCR fallback for washroom/exit.', e.message);
  }
}

function grabFrame(maxSide = 1280) {
  const { w, h } = frameSize();
  const s = Math.min(1, maxSide / Math.max(w, h));
  work.width = Math.round(w * s); work.height = Math.round(h * s);
  const c = work.getContext('2d', { willReadFrequently: true });
  c.drawImage(video, 0, 0, work.width, work.height);
  return work;
}

/** Run the custom sign classifier on a centre square crop (Teachable Machine preprocessing). */
function classifySign() {
  if (!S.signModel || !video.videoWidth) return null;
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(video);
    const [h, w] = img.shape;
    const s = Math.min(h, w);
    const crop = img.slice([Math.floor((h - s) / 2), Math.floor((w - s) / 2), 0], [s, s, 3]);
    const input = tf.image.resizeBilinear(crop, [224, 224]).toFloat().div(127.5).sub(1).expandDims(0);
    const probs = S.signModel.predict(input).dataSync();
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return { label: S.signLabels[best] || `class_${best}`, prob: probs[best] };
  });
}

// ================================================================ DRAWING: camera overlay
const COL = { amber: '#ffb020', teal: '#3fe0c5', coral: '#ff5a4e', paper: '#f4f6ff', ink: '#121838', muted: '#aab3dc' };
function colourOf(d, m) {
  if (S.mode === 'find' && d.class === S.findTarget) return COL.teal;
  if (OBSTACLES.has(d.class)) return m < 1 ? COL.coral : COL.amber;
  return COL.paper;
}
function drawOverlay(dets, sign) {
  const W = overlay.clientWidth, H = overlay.clientHeight;
  if (overlay.width !== W) overlay.width = W;
  if (overlay.height !== H) overlay.height = H;
  ctx.clearRect(0, 0, W, H);
  if (!S.showBoxes) return;
  const { w: vw, h: vh } = frameSize();
  const scale = Math.max(W / vw, H / vh), ox = (W - vw * scale) / 2, oy = (H - vh * scale) / 2;
  ctx.font = '700 17px "Atkinson Hyperlegible", system-ui, sans-serif';
  for (const d of dets) {
    let [x, y, w, h] = d.bbox;
    if (S.mirror) x = vw - x - w;
    const m = distanceOf(d), tr = d._track;
    const col = tr && (tr.approaching || tr.crossing) ? COL.coral : colourOf(d, m);
    const X = ox + x * scale, Y = oy + y * scale, Wd = w * scale, Hd = h * scale;
    // corner brackets instead of full boxes: less clutter over the camera image
    const k = Math.min(28, Wd / 3, Hd / 3);
    ctx.strokeStyle = col; ctx.lineWidth = 4; ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(X, Y + k); ctx.lineTo(X, Y); ctx.lineTo(X + k, Y);
    ctx.moveTo(X + Wd - k, Y); ctx.lineTo(X + Wd, Y); ctx.lineTo(X + Wd, Y + k);
    ctx.moveTo(X + Wd, Y + Hd - k); ctx.lineTo(X + Wd, Y + Hd); ctx.lineTo(X + Wd - k, Y + Hd);
    ctx.moveTo(X + k, Y + Hd); ctx.lineTo(X, Y + Hd); ctx.lineTo(X, Y + Hd - k);
    ctx.stroke();
    let label = `${niceName(d.class)}  ${m.toFixed(1)} m`;
    if (tr && tr.approaching) label += `  ⚠ ${tr.motion.ttc.toFixed(1)} s`;
    if (tr && tr.crossing) {                                   // sideways arrow through the object
      const dir = tr.crossing === 'right' ? 1 : -1, cy = Y + Hd / 2, cx = X + Wd / 2, L = Math.min(120, Wd);
      ctx.strokeStyle = COL.coral; ctx.lineWidth = 6; ctx.beginPath();
      ctx.moveTo(cx - dir * L / 2, cy); ctx.lineTo(cx + dir * L / 2, cy);
      ctx.lineTo(cx + dir * (L / 2 - 18), cy - 14); ctx.moveTo(cx + dir * L / 2, cy); ctx.lineTo(cx + dir * (L / 2 - 18), cy + 14);
      ctx.stroke();
    }
    if (tr && tr.approaching) {                                // pulsing outer frame
      ctx.strokeStyle = 'rgba(255, 90, 78, .6)'; ctx.lineWidth = 3;
      const pad = 8 + 6 * Math.sin(performance.now() / 120);
      ctx.strokeRect(X - pad, Y - pad, Wd + 2 * pad, Hd + 2 * pad);
    }
    const tw = ctx.measureText(label).width + 16;
    const ly = Math.max(0, Y - 30), lx = Math.max(0, Math.min(X, W - tw));
    ctx.fillStyle = col; roundRect(ctx, lx, ly, tw, 26, 6); ctx.fill();
    ctx.fillStyle = COL.ink; ctx.fillText(label, lx + 8, ly + 19);
  }
  if (S.mode === 'find') { // centre corridor: "straight ahead" zone
    const c0 = W * (0.5 - CFG.findCentre), c1 = W * (0.5 + CFG.findCentre);
    ctx.fillStyle = 'rgba(63, 224, 197, .08)'; ctx.fillRect(c0, 0, c1 - c0, H);
    ctx.strokeStyle = 'rgba(63, 224, 197, .55)'; ctx.lineWidth = 2; ctx.setLineDash([10, 10]);
    ctx.beginPath(); ctx.moveTo(c0, 0); ctx.lineTo(c0, H); ctx.moveTo(c1, 0); ctx.lineTo(c1, H); ctx.stroke();
    ctx.setLineDash([]);
  }
  if (sign && sign.label !== 'nothing' && sign.prob > 0.6) {
    const txt = `${SIGN_SPEECH[sign.label] || sign.label}  ${(sign.prob * 100).toFixed(0)}%`;
    ctx.font = '700 22px "Atkinson Hyperlegible", system-ui, sans-serif';
    const tw = ctx.measureText(txt).width + 24;
    ctx.fillStyle = '#1d4fd8'; roundRect(ctx, W - tw - 14, 14, tw, 40, 10); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.fillText(txt, W - tw - 2, 41);
  }
}
function roundRect(c, x, y, w, h, r) {
  c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
}

// ================================================================ DRAWING: radar (top-down view)
/** items: [{x (0..1 user view), m (metres), col, label}] */
function drawRadar(canvas, items, t, { labels = true } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!cw || !ch) return;
  if (canvas.width !== Math.round(cw * dpr)) { canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr); }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cw, ch);
  const half = (CFG.hfovDeg / 2) * Math.PI / 180;
  const ox = cw / 2, oy = ch - 8;
  const R = Math.min(ch - 16, (cw / 2 - 6) / Math.sin(half));
  const ang = x => (x - 0.5) * 2 * half;               // 0..1 → -half..+half
  const pt = (a, r) => [ox + Math.sin(a) * r, oy - Math.cos(a) * r];

  // field of view fan
  g.beginPath(); g.moveTo(ox, oy); g.arc(ox, oy, R, -Math.PI / 2 - half, -Math.PI / 2 + half); g.closePath();
  g.fillStyle = 'rgba(63, 224, 197, .06)'; g.fill();
  g.strokeStyle = 'rgba(170, 179, 220, .35)'; g.lineWidth = 1; g.stroke();
  // left / ahead / right zone dividers
  for (const zx of [CFG.leftZone, CFG.rightZone]) {
    const [x1, y1] = pt(ang(zx), R); g.beginPath(); g.moveTo(ox, oy); g.lineTo(x1, y1);
    g.setLineDash([4, 6]); g.stroke(); g.setLineDash([]);
  }
  // distance rings (every metre)
  g.font = '700 11px "Atkinson Hyperlegible", system-ui, sans-serif'; g.fillStyle = COL.muted;
  for (let r = 1; r <= CFG.radarRange; r++) {
    const rr = R * r / CFG.radarRange;
    g.beginPath(); g.arc(ox, oy, rr, -Math.PI / 2 - half, -Math.PI / 2 + half);
    g.strokeStyle = r === 1 ? 'rgba(255, 90, 78, .55)' : 'rgba(170, 179, 220, .22)'; g.stroke();
    if (labels && r < CFG.radarRange) { const [lx, ly] = pt(half, rr); g.fillText(`${r} m`, lx + 4, ly + 4); }
  }
  // sweep: a slow scan line that sweeps across the fan
  const s = Math.sin(t / 900) * half;
  const grad = g.createLinearGradient(ox, oy, ...pt(s, R));
  grad.addColorStop(0, 'rgba(63, 224, 197, .0)'); grad.addColorStop(1, 'rgba(63, 224, 197, .55)');
  g.strokeStyle = grad; g.lineWidth = 2; g.beginPath(); g.moveTo(ox, oy); g.lineTo(...pt(s, R)); g.stroke();
  // you
  g.fillStyle = COL.amber; g.beginPath(); g.arc(ox, oy, Math.max(3, Math.min(6, R * 0.025)), 0, Math.PI * 2); g.fill();
  // blips
  for (const it of items) {
    if (it.trail) {                                            // where it came from
      it.trail.forEach((p, i) => {
        const [tx, ty] = pt(ang(p.x), R * Math.min(p.m, CFG.radarRange) / CFG.radarRange);
        g.globalAlpha = 0.15 + 0.5 * i / it.trail.length; g.fillStyle = it.col;
        g.beginPath(); g.arc(tx, ty, 2.5, 0, Math.PI * 2); g.fill();
      });
      g.globalAlpha = 1;
    }
    const r = R * Math.min(it.m, CFG.radarRange) / CFG.radarRange;
    const [bx, by] = pt(ang(it.x), r);
    const pulse = 1 + 0.25 * Math.sin(t / 160);
    g.fillStyle = it.col; g.globalAlpha = 0.25;
    const dot = Math.max(3, Math.min(6, R * 0.025));
    g.beginPath(); g.arc(bx, by, dot * 2.2 * pulse, 0, Math.PI * 2); g.fill();
    g.globalAlpha = 1; g.beginPath(); g.arc(bx, by, dot, 0, Math.PI * 2); g.fill();
    if (labels && it.label) {
      g.font = '700 12px "Atkinson Hyperlegible", system-ui, sans-serif';
      g.fillStyle = COL.paper; g.fillText(it.label, bx + 10, by + 4);
    }
  }
}
function radarItems() {
  return S.lastDetections.map(d => {
    const m = distanceOf(d), tr = d._track;
    const moving = tr && (tr.approaching || tr.crossing);
    return { x: xFracOf(d), m, col: moving ? COL.coral : colourOf(d, m),
      label: niceName(d.class) + (tr && tr.approaching ? ' ⚠' : ''),
      trail: moving ? tr.hist.map(p => ({ x: p.x, m: distanceOf({ class: tr.cls, bbox: p.b }) })) : null };
  });
}
function radarLoop(t) {
  const c = $('radar');
  if (c && !$('app').hidden) drawRadar(c, radarItems(), t, { labels: c.clientWidth > 160 });
  requestAnimationFrame(radarLoop);
}
// Start screen: a slow, calm demo radar (the one orchestrated motion of the page)
function startRadarLoop(t) {
  if ($('startScreen').hidden) return;
  const demo = [
    { x: 0.22, m: 2.6 + Math.sin(t / 2400) * 0.4, col: COL.amber, label: 'chair' },
    { x: 0.55 + Math.sin(t / 3100) * 0.06, m: 1.6 + Math.cos(t / 2000) * 0.5, col: COL.coral, label: 'person' },
    { x: 0.78, m: 3.6, col: COL.teal, label: 'bottle' },
  ];
  drawRadar($('startRadar'), demo, t);
  requestAnimationFrame(startRadarLoop);
}

/** Big "nearest" readout on the camera */
function updateNearest(dets) {
  const box = $('nearest');
  const obs = dets.filter(d => OBSTACLES.has(d.class) || S.mode === 'find' && d.class === S.findTarget)
    .map(d => ({ d, m: distanceOf(d) })).sort((a, b) => a.m - b.m);
  if (!obs.length) { box.hidden = true; return; }
  const n = obs[0];
  box.hidden = false;
  $('nearestVal').textContent = `${n.m.toFixed(1)} m`;
  $('nearestLbl').textContent = `${niceName(n.d.class)}, ${clockOf(xFracOf(n.d))}`;
  box.classList.toggle('danger', n.m < 1);
}

// ================================================================ MAIN LOOP
let lastSignRun = 0, lastSign = null, loopTimer = null;
async function loop() {
  clearTimeout(loopTimer);
  if (!S.coco || !video.videoWidth) { loopTimer = setTimeout(loop, 300); return; }
  const t0 = performance.now();
  let dets = [];
  try { dets = await S.coco.detect(video, 12, CFG.minScore); } catch (e) { console.warn(e); }
  S.lastDetections = dets;
  updateTracks(dets, performance.now());

  if (S.signModel && t0 - lastSignRun > CFG.signEveryMs) {
    lastSignRun = t0;
    lastSign = classifySign();
    handleSign(lastSign);
  } else if (!S.signModel && S.mode === 'sign') {
    ocrSignScan(); // fallback: look for washroom / exit words with OCR
  }

  if (S.mode === 'guard') { motionWarnings(); guardStep(dets); }
  else if (S.mode === 'find') findStep(dets);
  else if (S.mode === 'sign') { beeper.stop(); signSearchStep(); }

  drawOverlay(dets, lastSign);
  updateNearest(dets);
  loopTimer = setTimeout(loop, Math.max(30, CFG.detectEveryMs - (performance.now() - t0)));
}

// ================================================================ MOTION: collision prediction
/* Most assistive apps describe ONE frame: "there is a person".
   Drishti follows every object across frames and asks: is it coming at me, and how soon?
   - Approaching: the object's size grows. For an object getting closer, size ∝ 1/distance, so
       growth rate g = d(ln size)/dt = speed / distance  →  time-to-collision = 1 / g.
     No depth sensor needed: just the rate at which the box grows.
   - Crossing: the object moves sideways across the view (important at road crossings).            */
const MOVERS = new Set(['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'dog', 'cat', 'cow', 'horse']);
let nextTrackId = 1;
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return i / (a[2] * a[3] + b[2] * b[3] - i || 1);
}
/** Match this frame's detections to existing tracks (same class, best overlap first). */
function updateTracks(dets, t) {
  const pairs = [];
  dets.forEach((d, di) => S.tracks.forEach((tr, ti) => {
    if (tr.cls !== d.class) return;
    const o = iou(tr.box, d.bbox);
    if (o > 0.2) pairs.push([o, di, ti]);
  }));
  pairs.sort((a, b) => b[0] - a[0]);
  const dUsed = new Set(), tUsed = new Set();
  const sample = d => ({ t, b: d.bbox, x: xFracOf(d) });
  for (const [, di, ti] of pairs) {
    if (dUsed.has(di) || tUsed.has(ti)) continue;
    dUsed.add(di); tUsed.add(ti);
    const tr = S.tracks[ti], d = dets[di];
    tr.box = d.bbox; tr.seen = t; tr.hist.push(sample(d)); d._track = tr;
  }
  dets.forEach((d, di) => {
    if (dUsed.has(di)) return;
    const tr = { id: nextTrackId++, cls: d.class, box: d.bbox, seen: t, hist: [sample(d)], lastWarn: 0, lastCross: 0 };
    S.tracks.push(tr); d._track = tr;
  });
  S.tracks = S.tracks.filter(tr => t - tr.seen < 700);          // forget objects gone for 0.7 s
  for (const tr of S.tracks) { tr.hist = tr.hist.filter(p => t - p.t < 1200); analyseTrack(tr, t); }
}
/** Least-squares slope of f(sample) over time, per second (robust to detector jitter). */
function slope(pts, f) {
  const n = pts.length, t0 = pts[0].t;
  let st = 0, sy = 0, stt = 0, sty = 0;
  for (const p of pts) { const x = (p.t - t0) / 1000, y = f(p); st += x; sy += y; stt += x * x; sty += x * y; }
  const den = n * stt - st * st;
  return den ? (n * sty - st * sy) / den : 0;
}
function analyseTrack(tr, t) {
  tr.motion = null; tr.approaching = false; tr.crossing = null;
  const h = tr.hist;
  if (h.length < 5 || h[h.length - 1].t - h[0].t < 500 || t - tr.seen > 250) return;
  const growth = slope(h, p => Math.log(Math.sqrt(p.b[2] * p.b[3])));   // relative size change per second
  const vx = slope(h, p => p.x);                                         // + = moving to the user's right
  const m = distanceOf({ class: tr.cls, bbox: tr.box });
  tr.motion = { growth, vx, ttc: growth > 0.05 ? 1 / growth : Infinity, m, x: h[h.length - 1].x };
  tr.approaching = growth > CFG.approachRate && tr.motion.ttc < CFG.ttcWarn && m < 6;
  tr.crossing = !tr.approaching && MOVERS.has(tr.cls) && Math.abs(vx) > CFG.crossRate && m < 8 ? (vx > 0 ? 'right' : 'left') : null;
}
/** Rising "whoosh" in the ear on the side the object is coming from */
function whoosh(x) {
  const a = audio(), t = a.currentTime;
  const o = a.createOscillator(), g = a.createGain();
  o.type = 'sawtooth';
  o.frequency.setValueAtTime(250, t); o.frequency.exponentialRampToValueAtTime(1400, t + 0.35);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.25, t + 0.05); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.4);
  o.connect(g);
  let pan = zoneOf(x) === 'left' ? -1 : zoneOf(x) === 'right' ? 1 : 0;
  if (S.swapEars) pan = -pan;
  const out = a.createChannelMerger(2); out.channelInterpretation = 'discrete';
  if (pan <= 0) g.connect(out, 0, 0);
  if (pan >= 0) g.connect(out, 0, 1);
  out.connect(a.destination);
  o.start(t); o.stop(t + 0.45);
}
/** Guard mode: speak about things that are COMING AT the user, or crossing in front. */
function motionWarnings() {
  const now = Date.now();
  // most urgent first
  const list = S.tracks.filter(tr => tr.motion).sort((a, b) => a.motion.ttc - b.motion.ttc);
  for (const tr of list) {
    const M = tr.motion, name = cap(niceName(tr.cls));
    if (tr.approaching && now - tr.lastWarn > 4000) {
      tr.lastWarn = now;
      S.seen.set(`${tr.cls}:${zoneOf(M.x)}`, now); S.lastGuardSpeak = now;   // Guard needn't repeat it
      whoosh(M.x); vibrate([60, 30, 60, 30, 60]);
      const secs = Math.max(1, Math.round(M.ttc));
      const z = zoneOf(M.x);
      say(MOVERS.has(tr.cls) ? 'approaching' : 'staticNear', [tr.cls, z, secs], { priority: 3 });
      setBadge(`${name} approaching: ${M.ttc.toFixed(1)} s`, true);
      return;
    }
    if (tr.crossing && now - tr.lastCross > 5000 && now - tr.lastWarn > 2000) {
      tr.lastCross = now; S.lastGuardSpeak = now;
      say('crossing', [tr.cls, tr.crossing === 'right'], { priority: 2 });
      setBadge(`${name} crossing →`.replace('→', tr.crossing === 'right' ? '→' : '←'), true);
      return;
    }
  }
}

// ---------------------------------------------------------------- GUARD
function guardStep(dets) {
  // Traffic lights and stop signs (known by the built-in model): announce once while in view
  const tNow = Date.now();
  for (const d of dets) {
    if (d.class !== 'traffic light' && d.class !== 'stop sign') continue;
    const k = 'signal:' + d.class;
    if (!S.seen.has(k) || tNow - S.seen.get(k) > 8000) {
      say(d.class === 'traffic light' ? 'trafficLight' : 'stopSignAhead', [], { priority: 3 });
    }
    S.seen.set(k, tNow);
  }
  const obstacles = dets.filter(d => OBSTACLES.has(d.class) || S.verbose)
    .map(d => ({ d, x: xFracOf(d), m: distanceOf(d) }));
  if (!obstacles.length) { beeper.stop(); setBadge('Guard: path clear'); return; }

  // Most urgent = closest, weighted towards the walking path (centre)
  const urgency = o => (1 / o.m) * (1.4 - Math.min(1, Math.abs(o.x - 0.5) * 2) * 0.6);
  obstacles.sort((a, b) => urgency(b) - urgency(a));
  const top = obstacles[0];

  // Directional beep: closer → faster & higher
  if (top.m < 4) {
    const interval = Math.round(Math.max(140, Math.min(1100, top.m * 330)));
    const freq = top.m < 1 ? 1100 : top.m < 2 ? 880 : 660;
    beeper.set(interval, top.x, freq, top.m < 1 ? 'square' : 'sine');
  } else beeper.stop();

  const now = Date.now();
  const urgent = top.m < 1.0 && zoneOf(top.x) === 'ahead';
  if (urgent) { vibrate([120, 60, 120]); setBadge('Stop! Obstacle ahead', true); }
  else setBadge('Guard');

  // CALM SPEECH: the beeps already carry direction + distance continuously.
  // Words are only for (a) something NEW appearing, or (b) an urgent "stop".
  // An object counts as "already announced" until it has been out of view for 5 s.
  const key = o => `${o.d.class}:${zoneOf(o.x)}`;
  for (const [k, t] of S.seen) if (now - t > 5000) S.seen.delete(k);         // forget things gone for 5 s
  for (const o of obstacles) if (S.seen.has(key(o))) S.seen.set(key(o), now); // still in view → keep quiet
  const isNew = !S.seen.has(key(top));

  if (urgent) {
    if (now - S.lastUrgent > 5000) {
      S.lastUrgent = now; S.seen.set(key(top), now);
      say('stopObj', [top.d.class], { priority: 3 });
    }
    return;
  }
  if (isNew && top.m < 3.5 && now - S.lastGuardSpeak > 3000) {
    S.lastGuardSpeak = now; S.seen.set(key(top), now);
    say('objAt', [top.d.class, zoneOf(top.x), top.m], { priority: 1 });
  }
}
function shortSteps(m) {
  if (m < 0.8) return 'very close';
  const s = Math.max(1, Math.round(m / CFG.stepMeters));
  return s === 1 ? '1 step' : `${s} steps`;
}

// ---------------------------------------------------------------- FIND (guided search)
function findGuidance(x, y, m) {
  const off = x - 0.5, side = off < 0 ? 'left' : 'right';
  const kind = Math.abs(off) < CFG.findCentre ? 'ahead' : Math.abs(off) < 0.25 ? 'slight' : 'side';
  const g = { kind, side };
  const vert = y < 0.3 ? 'up' : y > 0.72 ? 'down' : '';
  return { g, horiz: H.en.horiz(g), vert, side, centred: kind === 'ahead', steps: stepsText(m) };
}
function findStep(dets) {
  const T = S.track, now = Date.now(), W = cap(S.findWord);
  let cands = dets.filter(d => d.class === S.findTarget);
  // prefer the one closest to where we saw it last (stable tracking)
  if (cands.length > 1 && T.x != null) cands.sort((a, b) => Math.abs(xFracOf(a) - T.x) - Math.abs(xFracOf(b) - T.x));
  else cands.sort((a, b) => b.score - a.score);

  if (!cands.length) {
    // tolerate a few missed frames before calling it lost
    if (now - T.lastSeen < 1200) return;
    T.zone = null; T.stable = 0;
    beeper.set(1400, 0.5, 330, 'triangle', true); // slow, soft "searching" tick
    setBadge(`Find: searching for ${S.findWord}`);
    if (now - T.lastSpoke > 12000) {
      T.lastSpoke = now;
      if (T.lastSeen && now - T.lastSeen < 4000 && T.lastHoriz) say('lostIt', [T.lastHoriz]);
      else say('stillLooking');
    }
    return;
  }

  const d = cands[0], x = xFracOf(d), y = yFracOf(d), m = distanceOf(d);
  const G = findGuidance(x, y, m);
  const wasLost = !T.lastSeen || now - T.lastSeen > 2500;
  T.lastSeen = now; T.x = x; T.lastHoriz = G.g;

  // Geiger counter: more centred → faster. Closer → higher pitch. Side → left/right ear.
  const interval = Math.round(90 + Math.abs(x - 0.5) * 1000);
  const freq = m < 1 ? 1200 : m < 2 ? 950 : 750;
  beeper.set(interval, x, freq, 'sine');
  setBadge(`Find: ${S.findWord} ${G.horiz}, ${m.toFixed(1)} m`);

  // zone must be stable for 2 frames before we talk about a change
  const zone = G.centred ? 'centre' : G.side;
  T.stable = zone === T.zone ? T.stable + 1 : 1;
  T.zone = zone;

  // CALM SPEECH: speak only when something changes (found / turned the right way / within reach).
  // The beeps guide continuously in between; a short reminder at most every 10 s.
  if (wasLost) {
    T.lastSpoke = now; T.spokenZone = zone; chime(); vibrate([60, 40, 60]);
    say('found', [S.findTarget, G.g, G.vert, m], { priority: 3 });
    return;
  }
  if (G.centred && m < CFG.reachMeters) {
    if (!T.reach) {
      T.reach = true; T.lastSpoke = now; chime(); vibrate([200, 80, 400]);
      say('reach', [G.vert], { priority: 3 });
    }
    return;
  }
  if (m > CFG.reachMeters * 1.4) T.reach = false;

  const zoneChanged = T.stable === 3 && T.spokenZone !== zone;   // stable for 3 frames
  if (zoneChanged || now - T.lastSpoke > 10000) {
    T.lastSpoke = now; T.spokenZone = zone;
    if (G.centred) {
      vibrate([60, 40, 60]);
      say('straight', [m], { priority: 2 });
    } else {
      say('turn', [G.side], { priority: 1 });
    }
  }
}

// ---------------------------------------------------------------- SIGNS (custom model)
function handleSign(res) {
  if (!res) return;
  const { label, prob } = res;
  if (label === 'nothing' || prob < CFG.signThreshold) { S.signStreak = { label: null, n: 0 }; return; }
  S.signStreak = S.signStreak.label === label ? { label, n: S.signStreak.n + 1 } : { label, n: 1 };
  if (S.signStreak.n < CFG.signStableFrames) return;
  if (S.mode !== 'sign' && S.mode !== 'guard') return;          // stay quiet when idle / stopped
  if (S.mode === 'sign' && S.signTargets && !S.signTargets.includes(label)) return;
  const now = Date.now(), last = S.lastSignAnnounce.get(label) || 0;
  if (now - last < (S.mode === 'sign' ? 4000 : 9000)) return;
  S.lastSignAnnounce.set(label, now);
  chime(); vibrate([80, 50, 80]); signFound();
  say('sign', [label, ''], { priority: TRAFFIC_LABELS.includes(label) ? 3 : 2 });
}

// OCR fallback for washroom / exit when the custom model is not trained yet
async function ocrSignScan() {
  const now = Date.now();
  if (S.ocrScanBusy || now - S.lastOcrScan < 2500) return;
  S.ocrScanBusy = true; S.lastOcrScan = now;
  try {
    const text = (await runOcr(grabFrame(1000))).text.toUpperCase();
    const hit = matchSignWords(text);
    if (S.mode === 'sign' && hit && S.signTargets && S.signTargets.some(t => hit.labels.includes(t))) {   // still searching, and it's what we want
      chime(); vibrate([80, 50, 80]); signFound();
      say('sign', [hit.key, hit.dir], { priority: 2 });
    }
  } catch (e) { console.warn(e); }
  S.ocrScanBusy = false;
}
function matchSignWords(T) {
  const dir = /\bLEFT\b|←|<-/.test(T) ? 'left' : /\bRIGHT\b|→|->/.test(T) ? 'right' : /\bUP\b|↑|STRAIGHT/.test(T) ? 'straight ahead' : '';
  if (/MEN AT WORK|ROAD WORK|WORK IN PROGRESS|UNDER REPAIR|CONSTRUCTION|DIVERSION/.test(T)) return { key: 'road_work', say: signName('road_work'), labels: ['road_work'], dir };
  if (/\b(GENTS|MEN'?S?|MALE|BOYS)\b|पुरुष|ಪುರುಷ/.test(T)) return { key: 'washroom_men', say: "Men's washroom sign", labels: ['washroom_men'], dir };
  if (/\b(LADIES|WOMEN'?S?|FEMALE|GIRLS)\b|महिला|ಮಹಿಳೆ/.test(T)) return { key: 'washroom_women', say: "Women's washroom sign", labels: ['washroom_women'], dir };
  if (/TOILET|WASHROOM|RESTROOM|LAVATORY|W\.?C\b|शौचालय|ಶೌಚಾಲಯ/.test(T)) return { key: 'washroom', say: 'Washroom sign', labels: ['washroom_men', 'washroom_women'], dir };
  if (/\bEXIT\b|WAY OUT|निकास|ನಿರ್ಗಮನ/.test(T)) return { key: 'exit_sign', say: 'Exit sign', labels: ['exit_sign'], dir };
  if (/ZEBRA|CROSSING/.test(T)) return { key: 'pedestrian_crossing', say: signName('pedestrian_crossing'), labels: ['pedestrian_crossing'], dir };
  if (/\bSCHOOL\b/.test(T)) return { key: 'school_ahead', say: signName('school_ahead'), labels: ['school_ahead'], dir };
  if (/SPEED BREAKER|\bHUMP\b|GO SLOW/.test(T)) return { key: 'speed_breaker', say: signName('speed_breaker'), labels: ['speed_breaker'], dir };
  if (/NO ENTRY|DO NOT ENTER/.test(T)) return { key: 'no_entry', say: signName('no_entry'), labels: ['no_entry'], dir };
  if (/CAUTION|DANGER|WARNING/.test(T)) return { key: 'caution', say: signName('caution'), labels: ['caution'], dir };
  if (/^\W*STOP\W*$/.test(T)) return { key: 'stop', say: 'Stop sign', labels: ['stop'], dir };
  return null;
}

// ================================================================ OCR (Read mode)
async function getOcr() {
  const lang = $('setOcrLang').value;
  if (S.ocr && S.ocrLang === lang) return S.ocr;
  if (S.ocr) { await S.ocr.terminate(); S.ocr = null; }
  setCaption('Loading text reader…');
  // window.DRISHTI_OCR_OPTS lets you self-host worker/core/language files (fully offline build)
  S.ocr = await Tesseract.createWorker(lang, 1, window.DRISHTI_OCR_OPTS || {});
  S.ocrLang = lang;
  return S.ocr;
}
/**
 * Prepare a frame for OCR:
 *  1. keep only the CENTRE of the image (where the user points the camera) → no background junk
 *  2. upscale to ~1400 px wide so small text becomes readable
 *  3. grayscale + Otsu threshold → clean black-and-white text
 *  4. optional invert, for light text on dark signs (e.g. white EXIT on green)
 * The camera frame is never mirrored here, so text always reads the right way round.
 */
function preprocess(src, { invert = false, crop = 1, width = 1280, binarize = false } = {}) {
  const cw = src.width * crop, ch = src.height * crop;
  const sx = (src.width - cw) / 2, sy = (src.height - ch) / 2;
  const scale = width / cw;                      // zoom so the text lands at a size Tesseract likes
  const c = document.createElement('canvas');
  c.width = Math.round(cw * scale); c.height = Math.round(ch * scale);
  const g = c.getContext('2d', { willReadFrequently: true });
  g.imageSmoothingQuality = 'high';
  g.drawImage(src, sx, sy, cw, ch, 0, 0, c.width, c.height);
  const im = g.getImageData(0, 0, c.width, c.height), p = im.data;
  const hist = new Array(256).fill(0);
  for (let i = 0; i < p.length; i += 4) {
    const y = (0.299 * p[i] + 0.587 * p[i + 1] + 0.114 * p[i + 2]) | 0;
    p[i] = y; hist[y]++;
  }
  // Otsu: pick the threshold that best separates text from background
  const total = p.length / 4; let sum = 0; for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sB = 0, wB = 0, best = 0, thr = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (!wB) continue;
    const wF = total - wB; if (!wF) break;
    sB += t * hist[t];
    const mB = sB / wB, mF = (sum - sB) / wF, between = wB * wF * (mB - mF) ** 2;
    if (between > best) { best = between; thr = t; }
  }
  // contrast stretch (default) or hard black/white (binarize)
  let lo = 0, hi = 255, acc = 0;
  for (let t = 0; t < 256; t++) { acc += hist[t]; if (acc > total * 0.01) { lo = t; break; } }
  acc = 0; for (let t = 255; t >= 0; t--) { acc += hist[t]; if (acc > total * 0.01) { hi = t; break; } }
  const k = 255 / Math.max(1, hi - lo);
  for (let i = 0; i < p.length; i += 4) {
    let v = binarize ? (p[i] > thr ? 255 : 0) : Math.max(0, Math.min(255, (p[i] - lo) * k));
    if (invert) v = 255 - v;
    p[i] = p[i + 1] = p[i + 2] = v;
  }
  g.putImageData(im, 0, 0);
  return c;
}
/** Keep only real words: confident, and with at least 2 letters/digits (drops ". | & ~" junk) */
function cleanWords(data) {
  const out = [];
  for (const w of data.words || []) {
    const t = w.text.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
    const core = t.replace(/[^\p{L}\p{N}]/gu, '');
    if (w.confidence >= 60 && core.length >= 2) out.push({ t, conf: w.confidence });
    else if (w.confidence >= 80 && /^(\d|->|<-|[←→↑])$/.test(w.text)) out.push({ t: w.text, conf: w.confidence });
  }
  return out;
}
async function runOcr(canvas) {
  const w = await getOcr();
  // psm 11 = "sparse text": finds words anywhere (signs on a wall); psm 3 = normal page layout (documents)
  const attempt = async (psm, opts) => {
    await w.setParameters({ tessedit_pageseg_mode: psm });
    const { data } = await w.recognize(preprocess(canvas, opts));
    const words = cleanWords(data);
    return { text: words.map(x => x.t).join(' '), score: words.reduce((s, x) => s + x.t.length * x.conf, 0) };
  };
  // Tesseract is picky about letter size, so try several zoom levels and keep the best result.
  // Stops early as soon as it gets a confident word.
  const tries = [
    ['11', { width: 1400 }],                       // whole view
    ['11', { width: 1100 }],
    ['11', { crop: 0.6, width: 1700 }],            // zoom into the centre: small / far text
    ['11', { width: 850 }],                        // shrink: big close-up text
    ['3',  { width: 1400 }],                       // page layout: documents, labels
    ['11', { width: 1400, invert: true }],         // light text on a dark sign
  ];
  let best = { text: '', score: 0 };
  for (const [psm, opts] of tries) {
    const r = await attempt(psm, opts);
    if (r.score > best.score) best = r;
    if (best.score >= 300) break;
  }
  return { text: best.text.replace(/\s+/g, ' ').trim(), score: best.score };
}
async function readText() {
  if (S.busyRead) return;
  S.busyRead = true;
  const token = S.stopToken;               // if Stop is pressed while reading, stay silent
  const wasFinding = S.mode === 'find';
  beeper.stop(); vibrate(40);
  tone({ freq: 700, dur: 0.1 }); setCaption('Reading…');
  try {
    const { text } = await runOcr(grabFrame(1600));
    if (token !== S.stopToken) { S.busyRead = false; return; }
    if (!text || !/\p{L}{3,}/u.test(text)) {          // need at least one real word
      say('noText', [], { priority: 3 });
    } else {
      const hit = matchSignWords(text.toUpperCase());
      // a bare sign ("EXIT") → just name it; a longer notice → sign type first, then the full text as written
      if (hit && text.split(' ').length <= 2) say('sign', [hit.key, hit.dir], { priority: 3 });
      else if (hit) {                                   // one utterance: sign meaning, then the words as written
        const useLocal = S.lang === 'en' || hasVoice(S.lang);
        speak(`${msg(useLocal ? S.lang : 'en', 'sign', hit.key, hit.dir)} ${text}`, { priority: 3, lang: LANG_CODE[useLocal ? S.lang : 'en'] });
      }
      else speak(text, { priority: 3 });
    }
  } catch (e) {
    console.error(e);
    say('readerFail', [], { priority: 3 });
  }
  S.busyRead = false;
  if (wasFinding) S.track.lastSpoke = Date.now();
}

// ================================================================ DESCRIBE
async function describe() {
  if (!S.coco) { say('loading'); return; }
  vibrate(30);
  const dets = await S.coco.detect(video, 20, 0.45);
  const sign = S.signModel ? classifySign() : null;
  const parts = [];
  if (sign && sign.label !== 'nothing' && sign.prob > CFG.signThreshold) parts.push(signName(sign.label));
  if (!dets.length && !parts.length) { say('nothing', [], { priority: 3 }); return; }
  // group by object AND side, so a person on the left and a person on the right are reported separately
  const groups = {};
  for (const d of dets) {
    const k = `${d.class}|${zoneOf(xFracOf(d))}`;
    (groups[k] = groups[k] || []).push(d);
  }
  const items = Object.entries(groups).map(([k, arr]) => {
    arr.sort((a, b) => distanceOf(a) - distanceOf(b));
    return { cls: k.split('|')[0], n: arr.length, x: xFracOf(arr[0]), m: distanceOf(arr[0]) };
  }).sort((a, b) => a.x - b.x).slice(0, 4);               // read out left → right, max 4
  const signPart = parts.length ? msg(S.lang, 'sign', sign.label, '') + ' ' : '';
  const local = signPart + (items.length ? msg(S.lang, 'describe', items) : '');
  if (S.lang === 'en' || hasVoice(S.lang)) speak(local, { priority: 3, lang: LANG_CODE[S.lang] });
  else { speak((parts.length ? msg('en', 'sign', sign.label, '') + ' ' : '') + (items.length ? msg('en', 'describe', items) : ''), { priority: 3, lang: 'en-IN' }); $('caption').textContent = local; }
}

// ================================================================ MODES
function setBadge(t, alert = false) {
  if (!alert && Date.now() < S.badgeHoldUntil) return;   // keep an alert visible for a moment
  if (alert) S.badgeHoldUntil = Date.now() + 1500;
  const b = $('modeBadge'); b.textContent = t; b.classList.toggle('alert', alert);
}
function pressed(id, on) { $(id).setAttribute('aria-pressed', on ? 'true' : 'false'); }
function resetModes() {
  S.mode = 'idle'; S.findTarget = null; S.signTargets = null; S.signSearch = null; beeper.stop();
  ['btnGuard', 'btnFind', 'btnWashroom'].forEach(id => pressed(id, false));
  setBadge('Ready');
}
function startGuard() {
  resetModes(); S.mode = 'guard'; pressed('btnGuard', true); setBadge('Guard');
  S.seen.clear(); S.lastGuardSpeak = Date.now() - 1500; S.lastUrgent = 0;
  say('guardOn', [], { priority: 3 });
}
/** STOP: silences everything at once and stays silent until the user starts a mode again */
function stopAll(say = true) {
  S.stopToken = (S.stopToken || 0) + 1;     // cancels any reading in progress
  resetModes();
  silence();
  if (S.listening && rec) { try { rec.abort(); } catch (_) {} }
  if (say) { tone({ freq: 440, dur: 0.15 }); setCaption('Stopped.'); }
}
function startFind(word) {
  word = (word || '').toLowerCase().trim().replace(/^(my|the|a|an)\s+/, '');
  if (!word) { openFindDialog(); return; }
  const signKey = Object.keys(SIGN_TARGETS).find(k => word.includes(k));
  if (signKey) return startSignSearch(signKey);
  const key = Object.keys(FIND_ALIASES).find(k => word === k) || Object.keys(FIND_ALIASES).find(k => word.includes(k));
  const cls = key !== undefined ? FIND_ALIASES[key] : null;
  if (!cls) {
    say('cantTrack', [word], { priority: 2 });
    return;
  }
  resetModes(); S.mode = 'find'; S.findTarget = cls;
  S.findWord = key === 'water' || key === 'water bottle' ? 'bottle' : key === 'cell phone' || key === 'mobile' ? 'phone' : key;
  S.track = { x: null, lastSeen: 0, lastSpoke: Date.now(), zone: null, stable: 0, spokenZone: null, reach: false, lastHoriz: '' };
  pressed('btnFind', true); setBadge(`Find: ${S.findWord}`);
  say('finding', [S.findTarget], { priority: 3 });
}
function startSignSearch(key) {
  resetModes(); S.mode = 'sign'; S.signTargets = SIGN_TARGETS[key];
  const what = key === 'exit' ? 'exit' : key === 'traffic' ? 'road' : 'washroom';
  pressed('btnWashroom', what === 'washroom'); setBadge(`Looking for ${what} signs`);
  say('lookingFor', [what], { priority: 3 });
  S.signSearch = { what, start: Date.now(), lastFound: 0, lastNone: 0, noneCount: 0 };
}
/* While searching: if nothing has been found for a while, SAY SO, so the user isn't left guessing.
   8 s with no sign → "No washroom sign found." (repeats every 10 s, max 3 times, then stops searching). */
const NONE_FIRST_MS = 8000, NONE_REPEAT_MS = 10000, NONE_MAX = 3;
function signSearchStep() {
  const s = S.signSearch; if (!s) return;
  const now = Date.now(), quietSince = Math.max(s.start, s.lastFound);
  if (now - quietSince < NONE_FIRST_MS || now - s.lastNone < NONE_REPEAT_MS) return;
  s.lastNone = now; s.noneCount++;
  if (s.noneCount < NONE_MAX) {
    setBadge(`No ${s.what} sign found`);
    say('noneFound', [s.what], { priority: 2 });
  } else {
    say('noneHere', [s.what], { priority: 3 });
    resetModes(); setBadge(`No ${s.what} sign here`);
  }
}
function signFound() { if (S.signSearch) { S.signSearch.lastFound = Date.now(); S.signSearch.noneCount = 0; } }
function help() {
  say('help', [], { priority: 2 });
}

// ================================================================ VOICE COMMANDS
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null;
function listen() {
  if (!SR) { say('voiceNone', [], { priority: 2 }); return; }
  if (S.listening) { rec && rec.stop(); return; }
  window.speechSynthesis && speechSynthesis.cancel();
  beeper.stop();
  rec = new SR();
  rec.lang = S.lang === 'en' ? S.voiceLang : LANG_CODE[S.lang]; rec.interimResults = false; rec.maxAlternatives = 3;
  rec.onstart = () => { S.listening = true; $('btnMic').classList.add('listening'); tone({ freq: 1040, dur: 0.08 }); setCaption('Listening…'); };
  rec.onend = () => { S.listening = false; $('btnMic').classList.remove('listening'); };
  rec.onerror = (e) => { if (e.error !== 'no-speech' && e.error !== 'aborted') say('voiceFail', [e.error]); };
  rec.onresult = (e) => {
    const alts = Array.from(e.results[0]).map(a => a.transcript.toLowerCase().trim());
    handleCommand(alts[0], alts);
  };
  try { rec.start(); } catch (e) { console.warn(e); }
}
/* ------------------------------------------------------------------
   VOICE COMMAND TABLE (there is no database: the spoken sentence is matched
   against these keyword patterns, top to bottom; the first match wins).
   To add a phrase, add a word to the right pattern.
   ------------------------------------------------------------------ */
const OBJECT_WORDS = Object.keys(FIND_ALIASES).filter(k => FIND_ALIASES[k]).sort((a, b) => b.length - a.length);
const INTENTS = [
  { name: 'stop',     re: /\b(stop|pause|quiet|silence|cancel|enough|shut up|be quiet|stop talking)\b/, run: () => stopAll() },
  { name: 'washroom', re: /\b(washroom|wash room|toilet|restroom|rest room|bathroom|loo|lavatory|gents|ladies|urinal)\b/, run: () => startSignSearch('washroom') },
  { name: 'exit',     re: /\b(exit|way out|emergency|get out|outside)\b/, run: () => startSignSearch('exit') },
  { name: 'find',     re: /\b(find|where(?:'s| is| are)?|locate|search|look for|looking for|get me|i need|i lost|lost my|bring)\b/,
    run: t => {
      const obj = OBJECT_WORDS.find(w => new RegExp(`\\b${w}s?\\b`).test(t));
      if (obj) return startFind(obj);
      const m = t.match(/(?:find|where(?:'s| is| are)?|locate|search(?: for)?|look(?:ing)? for|get me|i need|lost(?: my)?)\s+(?:my |the |a |an )?(.+)/);
      return m ? startFind(m[1]) : openFindDialog();
    } },
  { name: 'traffic',  re: /\b(traffic|road sign|road signs|crossing|zebra|signal|cross the road|road)\b/, run: () => startSignSearch('traffic') },
  { name: 'swap',     re: /\b(swap|switch|reverse|wrong) (the )?(ears?|sides?|left and right)\b/, run: () => swapEars() },
  { name: 'read',     re: /\b(read|text|sign|signs|board|label|written|writing|say|says|name plate|notice|poster|price|medicine|letters?)\b/, run: () => readText() },
  { name: 'who',      re: /\b(who|anyone|anybody|someone|somebody|people|person|next to me|beside me|near me|with me)\b/, run: () => describe() },
  { name: 'describe', re: /\b(describe|what|around|see|look|scene|front|surround\w*|tell me|in front|nearby|room|this|that)\b/, run: () => describe() },
  { name: 'guard',    re: /\b(guard|walk|walking|navigate|obstacle|obstacles|start|go|move|path|safe|careful|clear)\b/, run: () => startGuard() },
  { name: 'help',     re: /\b(help|commands|what can you do|options)\b/, run: () => help() },
  { name: 'object',   re: new RegExp(`\\b(${OBJECT_WORDS.join('|')})s?\\b`), run: t => startFind(OBJECT_WORDS.find(w => new RegExp(`\\b${w}s?\\b`).test(t))) },
];
function handleCommand(text, alts = [text]) {
  setCaption(`“${text}”`);
  // try the best guess first, then the other things the speech engine thought it heard
  for (const t of [text, ...alts.slice(1)]) {
    let clean = t.toLowerCase().replace(/[?.!,।]/g, ' ').replace(/\s+/g, ' ').trim();
    const local = localCommand(clean);              // Hindi / Kannada words → English intent
    if (local) clean = local;
    const hit = INTENTS.find(i => i.re.test(clean));
    if (hit) { console.info('command:', hit.name, '←', clean); return hit.run(clean); }
  }
  say('tryCmds', [], { priority: 3 });
}

// ================================================================ GESTURES & UI
function setupGestures() {
  const st = $('stage');
  let pressTimer = null, lastTap = 0, tapTimer = null, longFired = false;
  st.addEventListener('pointerdown', () => {
    longFired = false;
    pressTimer = setTimeout(() => { longFired = true; vibrate(50); listen(); }, 600);
  });
  const cancel = () => clearTimeout(pressTimer);
  st.addEventListener('pointerleave', cancel);
  st.addEventListener('pointercancel', cancel);
  st.addEventListener('pointerup', () => {
    cancel();
    if (longFired) return;
    const now = Date.now();
    // a single tap while it is talking just stops the talking
    if (window.speechSynthesis && speechSynthesis.speaking && now - lastTap >= 320) {
      speechSynthesis.cancel(); speakingPriority = 0; lastTap = now; return;
    }
    if (now - lastTap < 320) { clearTimeout(tapTimer); lastTap = 0; readText(); }
    else { lastTap = now; tapTimer = setTimeout(describe, 330); }
  });
  st.addEventListener('contextmenu', e => e.preventDefault());
}
function openFindDialog() {
  const grid = $('findGrid'); grid.innerHTML = '';
  for (const w of FIND_MENU) {
    const b = document.createElement('button'); b.textContent = cap(w);
    b.onclick = () => { $('findDialog').close(); startFind(w); };
    grid.appendChild(b);
  }
  $('findDialog').showModal();
  say('whatFind', [], { priority: 3 });
}
function setupUI() {
  $('btnGuard').onclick = () => (S.mode === 'guard' ? stopAll() : startGuard());
  $('btnDescribe').onclick = describe;
  $('btnRead').onclick = readText;
  $('btnFind').onclick = () => (S.mode === 'find' ? stopAll() : openFindDialog());
  $('btnWashroom').onclick = () => (S.mode === 'sign' ? stopAll() : startSignSearch('washroom'));
  $('btnMic').onclick = listen;
  $('btnStop').onclick = () => stopAll();
  $('btnSoundTest').onclick = soundTest;
  // "Swap left and right" button, added next to the sound test (no HTML change needed)
  const sw = document.createElement('button');
  sw.id = 'btnSwapEars'; sw.className = 'ghost wide';
  const swLabel = () => { sw.textContent = S.swapEars ? 'Left and right: SWAPPED (tap to undo)' : 'Beeps in the wrong ear? Swap left and right'; };
  sw.onclick = () => { swapEars(); updateSwapLabel(); };
  $('btnSoundTest').after(sw); swLabel();
  $('btnSoundTest').textContent = 'Sound check: fix left and right';
  $('findCancel').onclick = () => $('findDialog').close();
  const toggleSettings = (open) => {
    const s = $('settings'); s.hidden = !open; $('btnSettings').setAttribute('aria-expanded', String(open));
    if (open) $('settingsDone').focus(); else $('btnSettings').focus();
  };
  $('btnSettings').onclick = () => toggleSettings($('settings').hidden);
  $('settingsDone').onclick = () => toggleSettings(false);
  $('setMirror').addEventListener('change', e => {
    setMirror(e.target.checked);
    try { localStorage.setItem('drishti.mirror', String(e.target.checked)); } catch (_) {}
  });
  const bind = (id, key, prop = 'value', apply = v => v) => {
    const el = $(id);
    try { const v = localStorage.getItem('drishti.' + key); if (v !== null) el[prop] = prop === 'checked' ? v === 'true' : v; } catch (_) {}
    const upd = () => { S[key] = apply(el[prop]); try { localStorage.setItem('drishti.' + key, String(el[prop])); } catch (_) {} };
    el.addEventListener('change', upd); el.addEventListener('input', upd); upd();
  };
  bind('setRate', 'rate', 'value', Number);
  bind('setOcrLang', 'ocrLangPref');
  bind('setVoiceLang', 'voiceLang');
  bind('setHaptics', 'haptics', 'checked');
  bind('setBoxes', 'showBoxes', 'checked');
  bind('setVerbose', 'verbose', 'checked');

  document.addEventListener('keydown', e => {
    if (S.calibrating && /^[lr]$/i.test(e.key)) { e.preventDefault(); setEar(e.key.toLowerCase() === 'l' ? 'left' : 'right'); return; }
    if ((e.key === 'x' || e.key === 'X') && !e.target.matches('textarea')) { e.preventDefault(); swapEars(); const b = $('btnSwapEars'); if (b) b.textContent = S.swapEars ? 'Left and right: SWAPPED (tap to undo)' : 'Beeps in the wrong ear? Swap left and right'; return; }
    if (e.target.matches('input, select, textarea') || e.metaKey || e.ctrlKey || $('findDialog').open) return;
    if ($('startScreen').hidden === false) return;
    const k = e.key.toUpperCase();
    const map = { G: () => (S.mode === 'guard' ? stopAll() : startGuard()), D: describe, R: readText,
      F: () => openFindDialog(), W: () => startSignSearch('washroom'), E: () => startSignSearch('exit'), T: () => startSignSearch('traffic'), X: () => swapEars(),
      V: listen, S: () => stopAll(), H: help, ' ': describe, ESCAPE: () => { $('settings').hidden = true; stopAll(); } };
    if (map[k]) { e.preventDefault(); map[k](); }
  });
}

// ================================================================ BOOT
async function boot() {
  $('startScreen').hidden = true;
  $('app').hidden = false;
  audio(); // unlock audio on user gesture
  requestAnimationFrame(radarLoop);
  setCaption('Starting the camera…');
  try {
    await startCamera();
  } catch (e) {
    console.error(e);
    say('camFail', [], { priority: 3 });
    return;
  }
  setupGestures();
  await loadModels();
  loop();
  let calibrated = false;
  try { calibrated = localStorage.getItem('drishti.swapEars') !== null; } catch (_) {}
  if (calibrated) say('ready', [], { priority: 3 });
  else soundTest();                                  // first run on this device: check left/right once
}

setupUI();
setupLanguageUI();
requestAnimationFrame(startRadarLoop);
$('startScreen').addEventListener('click', boot, { once: true });
if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1')) {
  navigator.serviceWorker.register('sw.js').catch(e => console.info('SW not registered', e));
}

// exposed for debugging in the console / tests
window.drishti = { S, handleCommand, clockOf, stepsText, matchSignWords, startFind, startGuard, stopAll, setMirror, zoneOf };
