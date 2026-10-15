# 👁️ Drishti — AI Eyes for the Visually Impaired

**HackAttack 2026 · Track: Sustainability & Social Impact**
*AI-Powered Environmental Awareness and Navigation for the Visually Impaired*

Drishti is a **free, voice-first web app**. Open it on any phone, point the camera, and it:

| Mode | How to start | What happens | Scenario from the problem statement |
|---|---|---|---|
| 🛡️ **Guard** | say "guard", or press **G** | Detects obstacles in real time. **Spatial audio**: the beep comes from the left or right ear depending on where the obstacle is, and gets faster as it gets closer. The phone vibrates when you're about to bump into something. | Obstacle awareness |
| 👁️ **Describe** | tap the screen, or say "describe" | "I see 2 chairs at 11 o'clock, about 3 steps away; a person at 1 o'clock…" (**clock-face directions**, the method used in orientation & mobility training) | Understanding surroundings |
| 📖 **Read** | double-tap, or say "read" | Reads signs, room labels, boards and medicine strips aloud, in English, Hindi or Kannada | Reading signs and text |
| 🚻 **Washroom / Exit** | say "find washroom" or "find exit" | **Our own trained model** (synthetic + open data) recognises washroom and exit signs and announces them. Falls back to OCR if the model isn't trained yet. | Finding a washroom, navigation |
| 🔍 **Find** | say "find bottle" | **Geiger-counter tone**: beeps get faster and higher as you point at the object, then "Bottle straight ahead, about 2 steps." | Everyday assistance |

✅ 100% free and open source · ✅ no accounts or API keys · ✅ AI runs **on the phone**, so no images leave the device · ✅ works offline after the first load

---

## 📁 What's in this folder

```
drishti/
├── README.md                ← you are here (step-by-step plan)
├── app/                     ← THE WEB APP (deploy this folder)
│   ├── index.html
│   ├── style.css
│   ├── app.js               ← all the logic, well commented
│   ├── sw.js                ← offline support
│   ├── manifest.json        ← "install as app"
│   ├── icons/
│   └── model/               ← put your Teachable Machine export here
├── ml/                      ← MODEL TRAINING
│   ├── generate_synthetic.py   ← synthetic dataset generator (domain randomization)
│   ├── crop_open_dataset.py    ← converts open-source sign datasets to our classes
│   └── requirements.txt
├── props/                   ← PRINT THESE for the live demo
├── docs/
│   ├── PITCH.md             ← slide-by-slide content
│   ├── DEMO_SCRIPT.md       ← the 3-minute demo, word for word
│   └── preview_grid.jpg     ← sample synthetic images for your slides
└── CREDITS.md               ← open-source licences (judges like this)
```

---

## 👥 Team roles

| Person | Owns | Steps |
|---|---|---|
| **A — ML** | dataset + custom model | 1, 2, 3 |
| **B — App** | running, testing and tweaking the app | 0, 4, 5 |
| **C — Pitch** | props, slides, demo video, rehearsal | 6, 7 (helps B test) |
| D (if you have one) | helps A collect real photos, then helps C | — |

---

## 🪜 Step-by-step

### Step 0 — Setup (10 min, everyone)
1. Install **Python 3** and **Chrome**. Install VS Code if you want to edit code.
2. Unzip this project.
3. Run the app on a laptop:
   ```bash
   cd drishti/app
   python -m http.server 8000
   ```
   Open **http://localhost:8000** in Chrome → click → allow camera and microphone.
   (The camera only works on `localhost` or `https://` — that's a browser security rule.)
4. Print the files in `props/` (or show them full-screen on a second phone).

> On first load the app downloads the AI libraries (~10 MB). Do this on good Wi-Fi.

### Step 1 — Generate the synthetic dataset (15 min, Person A)
```bash
cd drishti/ml
pip install -r requirements.txt
python generate_synthetic.py --per-class 200
```
This creates `ml/dataset/` with 4 folders (`washroom_men`, `washroom_women`, `exit_sign`, `nothing`) and `preview_grid.jpg`.

**Make it much better (strongly recommended, 10 min):** take 30–50 phone photos of *your college* — corridors, classroom walls, doors, notice boards — put them in a folder `ml/my_photos/`, and run:
```bash
python generate_synthetic.py --per-class 200 --bg-dir my_photos
```
The signs are then pasted onto real backgrounds from your venue, which makes the model far more accurate on demo day.

**What the generator randomises** (say this in the pitch — it's *domain randomization*): sign colour, text (MEN / GENTS / LADIES / TOILET / EXIT…), icon style, size, position, rotation, perspective tilt, lighting, glare, blur, motion blur, camera noise, partial occlusion, and JPEG quality. The `nothing` class includes **hard negatives** (ROOM 204, NO PARKING, LIBRARY…), so the model learns that not every sign is a washroom.

### Step 2 — (Optional) add open-source real images (15 min, Person A)
1. Go to **universe.roboflow.com**, search "exit sign" / "restroom sign" / "toilet sign".
2. Pick a dataset, check its **licence** (CC BY 4.0 = free, just give credit), and download it in **YOLOv8** format.
3. Convert and merge it into our folders:
   ```bash
   python crop_open_dataset.py --src path/to/download --out dataset --map "exit=exit_sign" "men=washroom_men" "women=washroom_women"
   ```
   (Use the class names that appear in that dataset's `data.yaml` on the left side of each `=`.)
4. Add the dataset name, author and licence to `CREDITS.md`.

You can also add 20–30 **phone photos of real signs** in your college straight into the class folders. Real photos + synthetic data = the best result.

### Step 3 — Train the model in Teachable Machine (15 min, Person A)
1. Open **https://teachablemachine.withgoogle.com/train/image** (free, no install).
2. Create **4 classes** with exactly these names: `washroom_men`, `washroom_women`, `exit_sign`, `nothing`.
3. For each class click **Upload** → choose all images in the matching `dataset/` folder.
4. Click **Train Model** (≈2–5 min). Test with the webcam preview using the printed props.
5. 📸 **Screenshot** "Advanced → Under the hood" (accuracy per class and the training curves) — for your slides.
6. **Export Model → Tensorflow.js → Download**. Unzip and copy `model.json`, `metadata.json` and `weights.bin` into `app/model/`.
7. Reload the app → the bottom bar shows **"sign model: 4 classes"** ✅

> Until this is done the app still works: washroom/exit search uses OCR to read the sign text.

### Step 4 — Test every mode (30 min, Person B + C)
Use headphones to hear left/right. Keyboard shortcuts on a laptop: **G** guard · **D** describe · **R** read · **F** find · **W** washroom · **E** exit · **V** voice · **S** stop · **H** help.

| Test | Expected |
|---|---|
| Settings → **Test left and right sound** | "Left" + low beep in the left ear, "Straight ahead" + double beep, "Right" + high beep in the right ear |
| Walk towards a chair in Guard mode | beeps speed up, "Chair straight ahead, 12 o'clock, about 2 steps away", vibration when very close |
| Tap the screen | spoken description with clock directions |
| Point at `props/prop_read_text.png` and double-tap | "Washroom sign, pointing right. It says: ROOM 204 …" |
| "Find bottle", sweep slowly | tone speeds up when centred → "Bottle straight ahead" |
| "Find washroom", point at the GENTS prop | chime + "Men's washroom sign ahead" |

**Directional sound:** left = low beep in the left ear, right = high beep in the right ear, straight ahead = double beep. Pitch differs by side, so it works even on laptop speakers; headphones make it clearest.
**Laptop webcam:** it faces you, so left and right are mirrored. The app detects this automatically; you can switch it in Settings → "Front camera".

Tune behaviour in the `CFG` block at the top of `app/app.js` (confidence thresholds, how often to speak, etc.).

### Step 5 — Put it on a phone with HTTPS (15 min, Person B)
Phones need **https** for the camera. The easiest free option is **GitHub Pages**:
1. Create a new **public** GitHub repo, e.g. `drishti`.
2. Upload **the contents of the `app/` folder** (index.html etc., not the folder itself) to the repo root.
3. Repo **Settings → Pages → Deploy from a branch → main / (root) → Save**.
4. After ~1 minute your app is live at `https://<username>.github.io/drishti/`. Open it on every team phone and "Add to Home Screen".

Every time you change a file, upload it again; the site updates in about a minute.

### Step 6 — Slides and a backup video (40 min, Person C)
- Use `docs/PITCH.md` (slide-by-slide content) and `docs/preview_grid.jpg`.
- **Record a backup demo video** on a phone (screen recording + the room) in case Wi-Fi or permissions fail on stage.

### Step 7 — Rehearse the demo (20 min, everyone)
Follow `docs/DEMO_SCRIPT.md`. Practise it twice with a blindfold. Charge the phones. Bring headphones and the printed props.

---

## ⏱️ Timeline (5 hours)

| Time | A (ML) | B (App) | C (Pitch) |
|---|---|---|---|
| 0:00–0:15 | Step 0 | Step 0 | Step 0, print props |
| 0:15–1:00 | Steps 1–2 (+ take college photos) | test app on a laptop, learn the code | collect real sign photos for A, outline slides |
| 1:00–1:30 | Step 3 (train + export) | Step 5 (deploy to GitHub Pages) | slides |
| 1:30–2:30 | retrain with more real photos if accuracy is low | Step 4 testing on phones, tune `CFG` | slides + screenshots |
| 2:30–3:30 | help test | polish, fix issues | record backup video |
| 3:30–4:30 | rehearse | rehearse | rehearse ×2 |
| 4:30–5:00 | buffer | buffer | buffer |

---

## 🧠 How it works (architecture)

```
 Phone camera ──► COCO-SSD (TensorFlow.js, on-device, 80 object types) ──┐
              │                                                          ├─► Decision layer ──► Speech (Web Speech API)
              ├─► Our sign classifier (MobileNet transfer learning,      │    • clock direction    ├─► Spatial beeps (Web Audio, stereo pan)
              │     trained on synthetic + open data)                    │    • distance (pinhole) └─► Vibration
              └─► Tesseract OCR (Eng / Hin / Kan) ───────────────────────┘    • rate-limiting
 Microphone ──► Voice commands (Web Speech API)
```
- **Distance** ≈ real object height ÷ (2·tan(FOV/2) · box-height fraction) — the pinhole camera model, using typical sizes (person 1.65 m, chair 0.9 m, bottle 0.25 m…).
- **Direction**: box centre → 10, 11, 12, 1 or 2 o'clock.
- **No speech spam**: each object class is announced at most every ~4.5 s, unless it's getting much closer or is urgent.

## ⚠️ Limitations (state these honestly — it's a judging criterion)
- The general object model knows **80 everyday classes** (person, chair, bottle, phone, laptop…), not doors, stairs, keys or potholes.
- Distance is an **estimate** from object size, not a depth sensor (±30–50%).
- The sign model is trained on a small, partly synthetic dataset; it works best on standard printed signs and needs more real data for production.
- Voice commands need internet in Chrome (Google speech service); everything else runs on-device.
- It is an **aid alongside a white cane**, not a replacement.

## 🚀 Future scope
Depth estimation (MiDaS) for true distance · stairs/door/pothole detection · indoor maps with QR/BLE beacons · full Hindi/Kannada voice interface · smart-glasses or ESP32-CAM wearable version · community-reported hazard map.
