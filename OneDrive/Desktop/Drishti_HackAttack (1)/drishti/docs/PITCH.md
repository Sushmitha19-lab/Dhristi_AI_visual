# Pitch deck — slide-by-slide content (6 slides, ~3 minutes + demo)

> Keep slides visual: one big idea per slide, very little text. Speak the details.

---

## Slide 1 — Title
**Drishti — AI eyes in every pocket**
Team name · members · HackAttack 2026 · Sustainability & Social Impact

*Say:* "India has the world's largest population of blind and visually impaired people — millions of them. Most can't afford a ₹30,000 assistive device. But almost all of them can access a smartphone."
*(Before the event, check and cite a source for any statistic you quote, e.g. WHO or the National Blindness & Visual Impairment Survey.)*

---

## Slide 2 — The problem
Four photos/icons: 🚧 obstacle · 🚻 washroom sign · 📄 printed label · 🔑 lost object
- Unfamiliar building → can't find the washroom or exit
- Obstacles at body height that a cane misses
- Signs, room numbers and labels are invisible
- Existing tools: expensive, special hardware, or cloud-only (privacy, data cost)

---

## Slide 3 — Our solution (show the app screenshot)
**One free web app. No hardware. No account. Works offline. Private.**
| 🛡️ Guard | 👁️ Describe | 📖 Read | 🚻 Signs | 🔍 Find |
|---|---|---|---|---|
| spatial-audio obstacle alerts | clock-face directions | Eng / Hindi / Kannada | our own trained model | Geiger-counter guidance |

Interaction designed for blind users: whole-screen gestures (tap / double-tap / hold), voice commands, sound + vibration, everything spoken.

---

## Slide 4 — How it works (architecture diagram from README)
- **On-device AI**: TensorFlow.js COCO-SSD (80 objects) + Tesseract OCR
- **Our custom model**: MobileNet transfer learning for washroom & exit signs
- **Synthetic data with domain randomization**: [show `preview_grid.jpg`]
  randomized colour, text, icons, angle, lighting, blur, noise, occlusion + hard negatives
  + open-source images (CC BY, credited) + photos from our own campus
- **Distance** from the pinhole camera model; **direction** as clock positions
- Show the Teachable Machine accuracy screenshot here

---

## Slide 5 — Impact & sustainability
- **Cost**: ₹0 — runs on any phone the user already owns (no e-waste, no new hardware)
- **Privacy**: images never leave the phone
- **Accessible**: installable, offline, multilingual
- **Scales**: any college, hospital, railway station or mall can add its own signs by generating synthetic data — no expensive data collection

---

## Slide 6 — Limitations & next steps (judges score this!)
**Honest limitations**: 80 object classes · distance is an estimate · small sign dataset · an aid, not a cane replacement
**Next**: depth estimation, stairs/door/pothole detection, indoor navigation with QR/BLE beacons, full regional-language voice, wearable (smart glasses / ESP32-CAM), testing with blind users via NGOs

**Closing line:** "Drishti means *vision*. We can't give sight back — but we can give independence, for free."
