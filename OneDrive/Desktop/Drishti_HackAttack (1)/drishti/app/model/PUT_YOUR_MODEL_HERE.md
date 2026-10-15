# Put your trained sign model in this folder

After training in Teachable Machine (see the main README, Step 2):

1. Click **Export Model** → **Tensorflow.js** tab → **Download** (not "Upload").
2. Unzip it. You get three files:
   - `model.json`
   - `metadata.json`
   - `weights.bin`
3. Copy all three **into this folder** (`app/model/`).
4. Reload the app. The bottom bar should change to **"sign model: 4 classes"**.

The class names in Teachable Machine **must be exactly**:

| Class name in Teachable Machine | What it is |
|---|---|
| `washroom_men` | men's / gents washroom signs |
| `washroom_women` | women's / ladies washroom signs |
| `exit_sign` | exit signs |
| `nothing` | everything else (backgrounds, other signs) |

Until the model is here, the app still works: "Find washroom / exit" falls back
to reading the sign text with OCR (words like GENTS, LADIES, TOILET, EXIT).
