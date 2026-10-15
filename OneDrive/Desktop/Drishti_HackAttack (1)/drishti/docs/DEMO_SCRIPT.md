# Live demo script (≈3 minutes)

## Before you go on stage ✅
- [ ] App open on the phone, already started (camera allowed), volume at max
- [ ] Phone connected to a **Bluetooth speaker** (so judges hear it) — or a laptop mirroring the phone screen
- [ ] Props placed: a **chair** in the walking path, **GENTS sign** taped on a wall/door, **text card** on a table, **water bottle** on a table
- [ ] Blindfold for the "user"
- [ ] Backup video ready on a laptop
- [ ] Sign model loaded (bottom bar says "sign model: 4 classes")

## Roles
- **User** (blindfolded) holds the phone at chest height, camera facing forward
- **Narrator** speaks to the judges
- **Helper** stands near the user for safety

---

### 0:00 — Set the scene (Narrator)
"Meet Rahul. He's visually impaired and visiting our college for the first time. All he has is his phone."
*(User puts on the blindfold.)*

### 0:15 — Guard mode: obstacle awareness
User: *press and hold the screen* → **"Guard"**
→ App: "Guard mode on…"
User walks slowly towards the chair.
→ Beeps get faster; **"Chair, 12 o'clock, about 2 steps away"** → **"Careful! chair… very close"** + vibration.
User steps around it.
Narrator: "Notice the beep comes from the left or right ear — the direction of the obstacle. Faster means closer. It's all running on the phone, with no internet needed."

### 0:50 — Washroom: our own trained model
User: *hold* → **"Find washroom"**
User turns slowly towards the wall.
→ Chime + **"Men's washroom sign ahead, 94 percent sure."**
Narrator: "This sign detector is a model *we trained today*, with synthetic data we generated using domain randomization — randomized lighting, angles, blur, even fake look-alike signs so it doesn't get fooled."

### 1:30 — Read mode
User points at the text card → *double tap*
→ **"Washroom sign, pointing right. It says: ROOM 204, AI and ML LAB, WASHROOM."**
Narrator: "It also reads Hindi and Kannada, in the right voice."

### 1:55 — Find mode: the wow moment
User: *hold* → **"Find bottle"**
User sweeps the phone slowly; the Geiger-counter tone speeds up…
→ **"Bottle straight ahead, about 2 steps."**
User walks forward and **picks up the bottle**. *(Pause here — let the judges react.)*

### 2:30 — Describe
User taps once → **"I see a person at 1 o'clock, 2 chairs at 11 o'clock…"**
Narrator: "Clock-face directions — the same method blind people learn in mobility training."

### 2:45 — Close (Narrator)
"Free. Private. Offline. On the phone Rahul already has. Drishti doesn't replace his cane — it gives him the information his cane can't."

---

## If something goes wrong
- **Voice not recognised** → use the on-screen buttons / keyboard keys (G, W, R, F, D)
- **Sign not detected** → use Read mode (double tap) on the sign — the OCR fallback announces "Men's washroom sign"
- **Camera or app frozen** → reload; if that fails, play the backup video and keep narrating
