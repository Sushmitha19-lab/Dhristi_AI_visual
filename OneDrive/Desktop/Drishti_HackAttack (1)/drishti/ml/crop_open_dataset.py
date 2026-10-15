"""
Drishti - turn an open-source DETECTION dataset into CLASSIFICATION images
==========================================================================

Free sign datasets (e.g. on Roboflow Universe, search "exit sign" or
"restroom sign") usually come in YOLO format:

    my_download/
        train/images/xxx.jpg
        train/labels/xxx.txt     <- lines: class_id cx cy w h   (normalised 0..1)
        data.yaml                <- names: ['exit', 'men', 'women', ...]

This script crops every labelled box (with some padding, so the model also sees
context) and saves it into our class folders, so it can be MERGED with the
synthetic data before uploading to Teachable Machine.

Usage:
    python crop_open_dataset.py --src my_download --out dataset \
        --map "exit=exit_sign" "men=washroom_men" "women=washroom_women"

Classes not mentioned in --map are skipped.
ALWAYS check the dataset's licence on its download page (CC BY 4.0 = give credit
in your slides/README) - note it in CREDITS.md.
"""

import argparse
import os
import random
import re

from PIL import Image


def read_names(src):
    for name in ("data.yaml", "data.yml"):
        path = os.path.join(src, name)
        if os.path.exists(path):
            text = open(path, encoding="utf-8").read()
            # names: ['a', 'b']   or   names:\n  0: a\n  1: b   or   names:\n- a\n- b
            m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
            if m:
                return [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
            block = text.split("names:", 1)[1] if "names:" in text else ""
            names = []
            for line in block.splitlines()[1:]:
                line = line.strip()
                if not line:
                    continue
                if re.match(r"^(\d+\s*:|-)", line):
                    names.append(re.sub(r"^(\d+\s*:|-)\s*", "", line).strip("'\""))
                else:
                    break
            if names:
                return names
    raise SystemExit("Could not find class names in data.yaml - pass the dataset's root folder as --src.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="root of the downloaded YOLO-format dataset")
    p.add_argument("--out", default="dataset")
    p.add_argument("--map", nargs="+", required=True, help='dataset_class=our_class, e.g. "exit=exit_sign"')
    p.add_argument("--pad", type=float, default=0.25, help="extra context around each box (fraction)")
    p.add_argument("--max-per-class", type=int, default=300)
    args = p.parse_args()

    names = read_names(args.src)
    mapping = dict(m.split("=", 1) for m in args.map)
    print("Dataset classes:", names)
    print("Mapping:", mapping)
    counts = {v: 0 for v in mapping.values()}

    label_files = []
    for root, _, files in os.walk(args.src):
        if os.path.basename(root) == "labels":
            label_files += [os.path.join(root, f) for f in files if f.endswith(".txt")]
    random.shuffle(label_files)

    for lf in label_files:
        img_dir = os.path.join(os.path.dirname(os.path.dirname(lf)), "images")
        stem = os.path.splitext(os.path.basename(lf))[0]
        img_path = next((os.path.join(img_dir, stem + e) for e in (".jpg", ".jpeg", ".png", ".webp")
                         if os.path.exists(os.path.join(img_dir, stem + e))), None)
        if not img_path:
            continue
        img = None
        for i, line in enumerate(open(lf).read().splitlines()):
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            if cid >= len(names) or names[cid] not in mapping:
                continue
            ours = mapping[names[cid]]
            if counts[ours] >= args.max_per_class:
                continue
            if img is None:
                img = Image.open(img_path).convert("RGB")
            cx, cy, w, h = (float(v) for v in parts[1:5])
            W, H = img.size
            w, h = w * (1 + args.pad), h * (1 + args.pad)
            box = (max(0, (cx - w / 2) * W), max(0, (cy - h / 2) * H),
                   min(W, (cx + w / 2) * W), min(H, (cy + h / 2) * H))
            if box[2] - box[0] < 16 or box[3] - box[1] < 16:
                continue
            os.makedirs(os.path.join(args.out, ours), exist_ok=True)
            img.crop(box).resize((224, 224)).save(os.path.join(args.out, ours, f"real_{stem}_{i}.jpg"), quality=92)
            counts[ours] += 1

    print("Cropped images per class:", counts)
    print("Remember to credit the dataset (name, author, licence) in CREDITS.md and your slides.")


if __name__ == "__main__":
    main()
