"""
Drishti - Synthetic training data generator
===========================================

Generates labelled images of the signs that matter most to visually impaired
users but that general-purpose models (COCO-SSD) do NOT know:

    washroom_men    - men's / gents restroom signs
    washroom_women  - women's / ladies restroom signs
    exit_sign       - EXIT / emergency exit signs
  Road / traffic signs (Indian / Vienna-convention style):
    pedestrian_crossing - red triangle, person walking on zebra stripes
    road_work           - red triangle, worker digging ("men at work")
    school_ahead        - red triangle, two children
    speed_breaker       - red triangle, road hump
    caution             - red triangle "!" or yellow CAUTION board
    no_entry            - red circle with white bar
    nothing         - backgrounds + "distractor" signs (hard negatives)

Technique: DOMAIN RANDOMIZATION
    Every image randomizes sign style, colours, text, icon shape, size, position,
    rotation, perspective tilt, lighting, blur, motion blur, sensor noise, glare,
    occlusion and JPEG compression. The model sees so much variety in synthetic
    data that real-world photos look like "just another variation".

Usage:
    pip install pillow numpy
    python generate_synthetic.py                  # 200 images per class -> ./dataset
    python generate_synthetic.py --per-class 300 --out dataset
    python generate_synthetic.py --bg-dir my_photos   # composite onto your own photos (recommended!)
    python generate_synthetic.py --props-only     # just make printable demo signs
    python generate_synthetic.py --classes washroom_men washroom_women exit_sign nothing   # only some classes

Outputs:
    dataset/<class>/*.jpg     -> upload each folder as one class in Teachable Machine
    preview_grid.jpg          -> contact sheet for your slides
    props/*.png               -> printable signs for the live demo
"""

import argparse
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

CLASSES = ["washroom_men", "washroom_women", "exit_sign",
           "pedestrian_crossing", "road_work", "school_ahead", "speed_breaker", "caution", "no_entry",
           "nothing"]
IMG_SIZE = 224  # Teachable Machine / MobileNet input size

# ---------------------------------------------------------------------------
# Fonts: try common fonts on Windows / macOS / Linux, fall back to Pillow's own
# ---------------------------------------------------------------------------
FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibrib.ttf", "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/impact.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]
FONTS = [f for f in FONT_CANDIDATES if os.path.exists(f)]


def get_font(size):
    if FONTS:
        try:
            return ImageFont.truetype(random.choice(FONTS), size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Text variants seen on real Indian & international signs
# ---------------------------------------------------------------------------
TEXT = {
    "washroom_men": ["MEN", "GENTS", "MALE", "GENTS TOILET", "MEN'S ROOM",
                     "BOYS", "TOILET", "RESTROOM", "WASHROOM", ""],
    "washroom_women": ["WOMEN", "LADIES", "FEMALE", "LADIES TOILET",
                       "WOMEN'S ROOM", "GIRLS", "TOILET", "RESTROOM", "WASHROOM", ""],
    "exit_sign": ["EXIT", "EXIT", "EXIT", "FIRE EXIT", "EMERGENCY EXIT", "WAY OUT", ""],
    "distractor": ["ROOM 204", "LAB 3", "OFFICE", "NO PARKING", "LIBRARY",
                   "CANTEEN", "PRINCIPAL", "STAFF ROOM", "PUSH", "PULL",
                   "NO SMOKING", "CSE DEPT", "SILENCE", "RECEPTION", "LIFT",
                   "STAIRS", "AUDITORIUM", "SEMINAR HALL", "ADMIN BLOCK"],
}

WASHROOM_PLATES = [(0, 70, 160), (20, 40, 90), (255, 255, 255), (30, 30, 30),
                   (180, 180, 180), (0, 110, 140), (120, 70, 40), (230, 230, 220)]
EXIT_PLATES = [(0, 140, 70), (0, 110, 50), (20, 170, 90), (200, 30, 30),
               (255, 255, 255), (10, 90, 40)]


def contrast_color(bg):
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    if lum > 150:
        return random.choice([(0, 0, 0), (20, 40, 120), (0, 100, 60), (180, 0, 0)])
    return random.choice([(255, 255, 255), (250, 250, 230), (255, 230, 0)])


def jitter(c, amt=25):
    return tuple(int(np.clip(v + random.randint(-amt, amt), 0, 255)) for v in c)


# ---------------------------------------------------------------------------
# Pictograms (drawn with simple shapes so we need no copyrighted artwork)
# ---------------------------------------------------------------------------
def draw_man(d, cx, top, h, color):
    head_r = h * 0.11
    d.ellipse([cx - head_r, top, cx + head_r, top + 2 * head_r], fill=color)
    body_top = top + 2.4 * head_r
    bw = h * 0.30
    # torso
    d.rounded_rectangle([cx - bw / 2, body_top, cx + bw / 2, body_top + h * 0.40],
                        radius=int(h * 0.05), fill=color)
    # arms
    aw = h * 0.07
    d.rounded_rectangle([cx - bw / 2 - aw - 2, body_top, cx - bw / 2 - 2, body_top + h * 0.36],
                        radius=int(aw / 2), fill=color)
    d.rounded_rectangle([cx + bw / 2 + 2, body_top, cx + bw / 2 + aw + 2, body_top + h * 0.36],
                        radius=int(aw / 2), fill=color)
    # legs
    lw = bw * 0.42
    leg_top = body_top + h * 0.38
    d.rectangle([cx - bw / 2, leg_top, cx - bw / 2 + lw, top + h], fill=color)
    d.rectangle([cx + bw / 2 - lw, leg_top, cx + bw / 2, top + h], fill=color)


def draw_woman(d, cx, top, h, color):
    head_r = h * 0.11
    d.ellipse([cx - head_r, top, cx + head_r, top + 2 * head_r], fill=color)
    body_top = top + 2.4 * head_r
    aw = h * 0.07
    sw = h * 0.16
    # arms
    d.rounded_rectangle([cx - sw - aw - 2, body_top, cx - sw - 2, body_top + h * 0.33],
                        radius=int(aw / 2), fill=color)
    d.rounded_rectangle([cx + sw + 2, body_top, cx + sw + aw + 2, body_top + h * 0.33],
                        radius=int(aw / 2), fill=color)
    # triangular dress (the key visual difference from the man icon)
    hem = body_top + h * 0.52
    d.polygon([(cx - sw, body_top), (cx + sw, body_top),
               (cx + h * 0.27, hem), (cx - h * 0.27, hem)], fill=color)
    # legs
    lw = h * 0.07
    d.rectangle([cx - h * 0.11, hem, cx - h * 0.11 + lw, top + h], fill=color)
    d.rectangle([cx + h * 0.11 - lw, hem, cx + h * 0.11, top + h], fill=color)


def draw_runner(d, x, top, h, color, flip=False):
    """Running figure for EXIT signs."""
    s = 1 if not flip else -1
    lw = max(3, int(h * 0.09))
    head_r = h * 0.11
    hx = x + s * h * 0.12
    d.ellipse([hx - head_r, top, hx + head_r, top + 2 * head_r], fill=color)
    neck = (x + s * h * 0.05, top + h * 0.28)
    hip = (x - s * h * 0.05, top + h * 0.60)
    d.line([neck, hip], fill=color, width=lw)
    d.line([neck, (x + s * h * 0.28, top + h * 0.42)], fill=color, width=lw)   # front arm
    d.line([neck, (x - s * h * 0.20, top + h * 0.38)], fill=color, width=lw)   # back arm
    knee = (x + s * h * 0.12, top + h * 0.78)
    d.line([hip, knee], fill=color, width=lw)
    d.line([knee, (x + s * h * 0.10, top + h)], fill=color, width=lw)           # front leg
    d.line([hip, (x - s * h * 0.28, top + h * 0.92)], fill=color, width=lw)     # back leg


def draw_arrow(d, x0, y0, w, h, color, direction):
    t = h * 0.3
    cy = y0 + h / 2
    if direction == "right":
        d.rectangle([x0, cy - t / 2, x0 + w * 0.6, cy + t / 2], fill=color)
        d.polygon([(x0 + w * 0.55, y0), (x0 + w, cy), (x0 + w * 0.55, y0 + h)], fill=color)
    elif direction == "left":
        d.rectangle([x0 + w * 0.4, cy - t / 2, x0 + w, cy + t / 2], fill=color)
        d.polygon([(x0 + w * 0.45, y0), (x0, cy), (x0 + w * 0.45, y0 + h)], fill=color)
    else:  # up
        cx = x0 + w / 2
        d.rectangle([cx - t / 2, y0 + h * 0.4, cx + t / 2, y0 + h], fill=color)
        d.polygon([(x0, y0 + h * 0.45), (cx, y0), (x0 + w, y0 + h * 0.45)], fill=color)


def fit_text(d, text, max_w, max_h):
    size = int(max_h)
    while size > 8:
        f = get_font(size)
        bbox = d.textbbox((0, 0), text, font=f)
        if bbox[2] - bbox[0] <= max_w and bbox[3] - bbox[1] <= max_h:
            return f, bbox
        size = int(size * 0.85)
    f = get_font(8)
    return f, d.textbbox((0, 0), text, font=f)


def draw_text_centered(d, text, box, color):
    x0, y0, x1, y1 = box
    f, bb = fit_text(d, text, x1 - x0, y1 - y0)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text((x0 + (x1 - x0 - tw) / 2 - bb[0], y0 + (y1 - y0 - th) / 2 - bb[1]), text, font=f, fill=color)


# ---------------------------------------------------------------------------
# Sign renderers - each returns an RGBA image of the sign alone
# ---------------------------------------------------------------------------
def render_washroom(gender, W=None, H=None):
    W = W or random.randint(260, 420)
    layout = random.choice(["icon_text_below", "icon_left_text", "icon_only", "text_only_with_icon_small"])
    H = H or (random.randint(300, 460) if layout != "icon_left_text" else random.randint(150, 230))
    bg = jitter(random.choice(WASHROOM_PLATES), 15)
    fg = contrast_color(bg)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=random.randint(0, 30), fill=bg + (255,))
    if random.random() < 0.5:  # border
        d.rounded_rectangle([6, 6, W - 7, H - 7], radius=random.randint(0, 24), outline=fg, width=random.randint(2, 6))

    draw = draw_man if gender == "men" else draw_woman
    text = random.choice(TEXT["washroom_" + gender])
    if layout == "icon_only" or not text:
        draw(d, W / 2, H * 0.1, H * 0.8, fg)
    elif layout == "icon_text_below":
        draw(d, W / 2, H * 0.07, H * 0.6, fg)
        draw_text_centered(d, text, (W * 0.08, H * 0.72, W * 0.92, H * 0.93), fg)
    elif layout == "icon_left_text":
        draw(d, W * 0.2, H * 0.1, H * 0.8, fg)
        draw_text_centered(d, text, (W * 0.38, H * 0.25, W * 0.95, H * 0.75), fg)
    else:
        draw_text_centered(d, text, (W * 0.08, H * 0.08, W * 0.92, H * 0.45), fg)
        draw(d, W / 2, H * 0.5, H * 0.42, fg)
    if random.random() < 0.3:  # direction arrow under/next to it
        draw_arrow(d, W * 0.72, H * 0.80, W * 0.2, H * 0.12, fg, random.choice(["left", "right", "up"]))
    return img


def render_exit(W=None, H=None):
    W = W or random.randint(320, 520)
    H = H or random.randint(130, 230)
    bg = jitter(random.choice(EXIT_PLATES), 15)
    fg = contrast_color(bg) if bg != (255, 255, 255) else random.choice([(0, 140, 70), (200, 30, 30)])
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=random.randint(0, 18), fill=bg + (255,))
    text = random.choice(TEXT["exit_sign"])
    flip = random.random() < 0.5
    layout = random.random()
    if layout < 0.45:  # runner + text (+ arrow)
        rx = W * (0.82 if flip else 0.15)
        draw_runner(d, rx, H * 0.12, H * 0.76, fg, flip=flip)
        tb = (W * 0.05, H * 0.15, W * 0.68, H * 0.85) if flip else (W * 0.32, H * 0.15, W * 0.95, H * 0.85)
        draw_text_centered(d, text or "EXIT", tb, fg)
    elif layout < 0.75:  # text only
        draw_text_centered(d, text or "EXIT", (W * 0.06, H * 0.1, W * 0.94, H * 0.9), fg)
    else:  # ISO style: runner + door + arrow, no text
        draw_runner(d, W * 0.35, H * 0.12, H * 0.76, fg, flip=flip)
        d.rectangle([W * 0.08, H * 0.12, W * 0.22, H * 0.88], outline=fg, width=max(3, int(H * 0.05)))
        draw_arrow(d, W * 0.55, H * 0.3, W * 0.35, H * 0.4, fg, "left" if flip else "right")
    return img


def render_distractor():
    """Hard negatives: signs that are NOT washrooms/exits, so the model learns the difference."""
    W, H = random.randint(260, 520), random.randint(120, 260)
    bg = jitter(random.choice(WASHROOM_PLATES + EXIT_PLATES + [(240, 200, 0), (200, 30, 30)]), 20)
    fg = contrast_color(bg)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=random.randint(0, 20), fill=bg + (255,))
    draw_text_centered(d, random.choice(TEXT["distractor"]), (W * 0.06, H * 0.15, W * 0.94, H * 0.85), fg)
    if random.random() < 0.3:
        draw_arrow(d, W * 0.75, H * 0.65, W * 0.2, H * 0.25, fg, random.choice(["left", "right", "up"]))
    return img


# ---------------------------------------------------------------------------
# Road / traffic signs (pictograms drawn from simple shapes)
# ---------------------------------------------------------------------------
TRAFFIC_TEXT = {
    "pedestrian_crossing": ["ZEBRA CROSSING", "PEDESTRIAN CROSSING", "CROSSING", ""],
    "road_work": ["MEN AT WORK", "ROAD WORK", "WORK IN PROGRESS", "ROAD UNDER REPAIR", ""],
    "school_ahead": ["SCHOOL", "SCHOOL AHEAD", "SCHOOL ZONE", "GO SLOW", ""],
    "speed_breaker": ["SPEED BREAKER", "HUMP", "GO SLOW", ""],
    "caution": ["CAUTION", "DANGER", "WARNING", "BE CAREFUL", ""],
}


def lw(h, f=0.07):
    return max(3, int(h * f))


def glyph_walker(d, cx, top, h, col):
    """Person walking (side view)."""
    w = lw(h, 0.09)
    d.ellipse([cx - h * .07, top, cx + h * .09, top + h * .16], fill=col)
    neck, hip = (cx, top + h * .2), (cx - h * .03, top + h * .55)
    d.line([neck, hip], fill=col, width=w)
    d.line([neck, (cx + h * .16, top + h * .45)], fill=col, width=w)
    d.line([neck, (cx - h * .14, top + h * .42)], fill=col, width=w)
    d.line([hip, (cx + h * .15, top + h * .78), (cx + h * .2, top + h)], fill=col, width=w, joint="curve")
    d.line([hip, (cx - h * .12, top + h * .8), (cx - h * .2, top + h * .97)], fill=col, width=w, joint="curve")


def glyph_child(d, cx, top, h, col):
    w = lw(h, 0.1)
    d.ellipse([cx - h * .1, top, cx + h * .1, top + h * .2], fill=col)
    d.line([(cx, top + h * .22), (cx, top + h * .62)], fill=col, width=w)
    d.line([(cx - h * .15, top + h * .45), (cx, top + h * .28), (cx + h * .15, top + h * .45)], fill=col, width=w)
    d.line([(cx, top + h * .6), (cx - h * .12, top + h)], fill=col, width=w)
    d.line([(cx, top + h * .6), (cx + h * .12, top + h)], fill=col, width=w)


def draw_pictogram(kind, d, box, col):
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    if kind == "pedestrian_crossing":
        glyph_walker(d, cx, y0, H * .72, col)
        for i in range(5):                                   # zebra stripes
            sx = x0 + W * (0.08 + i * 0.19)
            d.polygon([(sx, y1 - H * .2), (sx + W * .1, y1 - H * .2), (sx + W * .06, y1), (sx - W * .04, y1)], fill=col)
    elif kind == "road_work":
        w = lw(H, 0.08)
        hx, hy = cx - W * .12, y0 + H * .05                  # bent worker with a shovel
        d.ellipse([hx - H * .08, hy, hx + H * .08, hy + H * .16], fill=col)
        hip = (cx - W * .02, y0 + H * .58)
        d.line([(hx + H * .03, hy + H * .18), hip], fill=col, width=w)
        d.line([hip, (cx - W * .14, y1)], fill=col, width=w)
        d.line([hip, (cx + W * .06, y1)], fill=col, width=w)
        d.line([(hx + H * .05, hy + H * .25), (cx + W * .18, y0 + H * .55)], fill=col, width=w)   # arms
        d.line([(cx - W * .05, y0 + H * .2), (cx + W * .26, y0 + H * .8)], fill=col, width=max(2, w // 2))  # shovel
        d.pieslice([cx + W * .1, y0 + H * .62, cx + W * .5, y1 + H * .35], 180, 360, fill=col)     # pile
    elif kind == "school_ahead":
        glyph_child(d, cx - W * .16, y0 + H * .08, H * .92, col)
        glyph_child(d, cx + W * .16, y0 + H * .25, H * .75, col)
        d.rectangle([cx - W * .3, y0 + H * .35, cx - W * .22, y0 + H * .5], fill=col)               # school bag
    elif kind == "speed_breaker":
        base = y1 - H * .12
        d.rectangle([x0 + W * .05, base, x1 - W * .05, base + H * .1], fill=col)
        for hx in (cx - W * .22, cx + W * .22):
            d.pieslice([hx - W * .18, base - H * .35, hx + W * .18, base + H * .35], 180, 360, fill=col)
    elif kind == "caution":
        bw = W * .14
        d.rounded_rectangle([cx - bw / 2, y0, cx + bw / 2, y0 + H * .68], radius=int(bw / 2), fill=col)
        d.ellipse([cx - bw * .6, y1 - bw * 1.2, cx + bw * .6, y1], fill=col)


def render_traffic(kind):
    S = random.randint(300, 460)
    red = jitter((200, 25, 30), 25)
    yellow_variant = kind == "caution" and random.random() < 0.35
    inner = jitter(random.choice([(255, 255, 255), (250, 250, 240), (255, 220, 0)]), 8) if random.random() < 0.25 else jitter((250, 250, 248), 6)
    ink = (15, 15, 15)
    text = random.choice(TRAFFIC_TEXT.get(kind, [""])) if random.random() < 0.35 else ""
    plate_h = int(S * 0.28) if text else 0
    img = Image.new("RGBA", (S, S + plate_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if kind == "no_entry":
        d.ellipse([0, 0, S - 1, S - 1], fill=red)
        d.rectangle([S * .16, S * .4, S * .84, S * .6], fill=(255, 255, 255))
    elif yellow_variant:                                    # yellow "CAUTION" board / diamond
        y = jitter((255, 200, 0), 15)
        if random.random() < 0.5:
            d.polygon([(S / 2, 0), (S - 1, S / 2), (S / 2, S - 1), (0, S / 2)], fill=y, outline=ink, width=lw(S, .03))
            draw_pictogram("caution", d, (S * .38, S * .22, S * .62, S * .78), ink)
        else:
            d.rectangle([0, S * .2, S - 1, S * .8], fill=y)
            d.polygon([(S * .2, S * .28), (S * .32, S * .5), (S * .08, S * .5)], fill=ink)
            draw_text_centered(d, random.choice(["CAUTION", "DANGER", "WARNING"]), (S * .36, S * .3, S * .95, S * .7), ink)
    else:                                                   # red-bordered warning triangle
        h = S * 0.87
        top = (S - h) / 2
        d.polygon([(S / 2, top), (S - 1, top + h), (0, top + h)], fill=red)
        b = S * random.uniform(0.11, 0.15)
        d.polygon([(S / 2, top + b * 2), (S - 1 - b * 1.75, top + h - b), (b * 1.75, top + h - b)], fill=inner)
        draw_pictogram(kind, d, (S * .3, top + h * .36, S * .7, top + h * .86), ink)
    if text:
        d.rectangle([S * .05, S + 4, S * .95, S + plate_h - 2], fill=(255, 255, 255), outline=ink, width=3)
        draw_text_centered(d, text, (S * .1, S + 8, S * .9, S + plate_h - 6), ink)
    if random.random() < 0.25:                              # sign post
        post = Image.new("RGBA", (img.width, img.height + S // 2), (0, 0, 0, 0))
        pd = ImageDraw.Draw(post)
        pd.rectangle([S * .46, img.height - 2, S * .54, post.height], fill=jitter((120, 120, 120), 20))
        post.paste(img, (0, 0), img)
        img = post
    return img


def render_road_distractor():
    """Hard negatives that look like road signs but aren't our classes."""
    S = random.randint(260, 420)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    k = random.random()
    if k < 0.35:   # blue round mandatory sign with an arrow
        d.ellipse([0, 0, S - 1, S - 1], fill=jitter((20, 70, 170), 15))
        draw_arrow(d, S * .2, S * .3, S * .6, S * .4, (255, 255, 255), random.choice(["left", "right", "up"]))
    elif k < 0.65:  # speed limit
        d.ellipse([0, 0, S - 1, S - 1], fill=jitter((200, 25, 30), 20))
        d.ellipse([S * .13, S * .13, S * .87, S * .87], fill=(255, 255, 255))
        draw_text_centered(d, str(random.choice([20, 30, 40, 50, 60])), (S * .25, S * .3, S * .75, S * .7), (0, 0, 0))
    else:           # blue / green direction board
        d.rectangle([0, S * .2, S - 1, S * .8], fill=jitter(random.choice([(20, 70, 170), (0, 120, 60)]), 15))
        draw_text_centered(d, random.choice(["BENGALURU 12", "HOSPITAL", "PARKING", "BUS STOP", "METRO"]), (S * .08, S * .3, S * .92, S * .7), (255, 255, 255))
    return img


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------
def procedural_background(size):
    w, h = size
    kind = random.random()
    base = np.array([random.randint(60, 240) for _ in range(3)], dtype=np.float32)
    if kind < 0.35:  # painted wall with gradient lighting
        gy = np.linspace(random.uniform(0.6, 1.0), random.uniform(0.7, 1.15), h)[:, None, None]
        gx = np.linspace(random.uniform(0.7, 1.1), random.uniform(0.7, 1.1), w)[None, :, None]
        arr = base[None, None, :] * gy * gx
    elif kind < 0.55:  # tiles
        arr = np.ones((h, w, 3), np.float32) * base
        step = random.randint(20, 70)
        arr[::step, :, :] *= 0.6
        arr[:, ::step, :] *= 0.6
    elif kind < 0.75:  # corridor: floor + wall + door shapes
        arr = np.ones((h, w, 3), np.float32) * base
        horizon = int(h * random.uniform(0.55, 0.8))
        arr[horizon:] = np.array([random.randint(40, 180) for _ in range(3)], np.float32)
        for _ in range(random.randint(0, 3)):
            x0 = random.randint(0, w - 30)
            dw = random.randint(30, w // 3)
            arr[int(h * 0.2):horizon, x0:x0 + dw] = np.array([random.randint(60, 200) for _ in range(3)], np.float32)
    else:  # noisy random blobs (clutter)
        arr = np.ones((h, w, 3), np.float32) * base
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        for _ in range(random.randint(5, 25)):
            x0, y0 = random.randint(0, w), random.randint(0, h)
            x1, y1 = x0 + random.randint(10, w // 2), y0 + random.randint(10, h // 2)
            col = tuple(random.randint(0, 255) for _ in range(3))
            (d.rectangle if random.random() < 0.5 else d.ellipse)([x0, y0, x1, y1], fill=col)
        arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0, random.uniform(2, 10), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class Backgrounds:
    def __init__(self, bg_dir):
        self.files = []
        if bg_dir and os.path.isdir(bg_dir):
            for root, _, files in os.walk(bg_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                        self.files.append(os.path.join(root, f))
            print(f"Using {len(self.files)} real background photos from {bg_dir}")

    def get(self, size):
        if self.files and random.random() < 0.7:
            try:
                img = Image.open(random.choice(self.files)).convert("RGB")
                # random crop
                s = random.uniform(0.4, 1.0)
                cw, ch = int(img.width * s), int(img.height * s)
                x, y = random.randint(0, img.width - cw), random.randint(0, img.height - ch)
                return img.crop((x, y, x + cw, y + ch)).resize(size)
            except Exception:
                pass
        return procedural_background(size)


# ---------------------------------------------------------------------------
# Augmentations (the "randomization" in domain randomization)
# ---------------------------------------------------------------------------
def perspective_coeffs(src, dst):
    m = []
    for (x, y), (u, v) in zip(dst, src):
        m.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        m.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    A = np.array(m, dtype=np.float64)
    B = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, B).tolist()


def random_perspective(sign, strength=0.18):
    w, h = sign.size
    dx, dy = w * strength, h * strength
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(random.uniform(0, dx), random.uniform(0, dy)),
           (w - random.uniform(0, dx), random.uniform(0, dy)),
           (w - random.uniform(0, dx), h - random.uniform(0, dy)),
           (random.uniform(0, dx), h - random.uniform(0, dy))]
    return sign.transform((w, h), Image.PERSPECTIVE, perspective_coeffs(src, dst), Image.BICUBIC)


def motion_blur(img, k):
    arr = np.asarray(img).astype(np.float32)
    out = np.zeros_like(arr)
    horizontal = random.random() < 0.5
    for i in range(k):
        out += np.roll(arr, i - k // 2, axis=1 if horizontal else 0)
    return Image.fromarray(np.clip(out / k, 0, 255).astype(np.uint8))


def add_glare(img):
    w, h = img.size
    arr = np.asarray(img).astype(np.float32)
    cx, cy = random.uniform(0, w), random.uniform(0, h)
    r = random.uniform(0.15, 0.5) * w
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))[..., None]
    arr = arr + mask * random.uniform(60, 160)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def compose(sign, bgs, min_scale=0.35, max_scale=0.95):
    canvas_size = (IMG_SIZE * 2, IMG_SIZE * 2)  # work at 2x then downscale (anti-aliasing)
    bg = bgs.get(canvas_size)
    if sign is not None:
        if random.random() < 0.8:
            sign = random_perspective(sign)
        sign = sign.rotate(random.uniform(-15, 15), resample=Image.BICUBIC, expand=True)
        scale = random.uniform(min_scale, max_scale) * canvas_size[0] / max(sign.size)
        sign = sign.resize((max(8, int(sign.width * scale)), max(8, int(sign.height * scale))), Image.BICUBIC)
        # lighting on the sign itself
        rgb = sign.convert("RGB")
        rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.55, 1.35))
        rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.6, 1.3))
        sign = Image.merge("RGBA", (*rgb.split(), sign.split()[3]))
        max_x = max(0, canvas_size[0] - sign.width)
        max_y = max(0, canvas_size[1] - sign.height)
        x, y = random.randint(0, max_x), random.randint(0, max_y)
        if random.random() < 0.4:  # drop shadow
            shadow = Image.new("RGBA", sign.size, (0, 0, 0, 0))
            shadow.putalpha(sign.split()[3].point(lambda a: int(a * 0.35)))
            bg.paste(shadow, (x + random.randint(2, 10), y + random.randint(2, 10)), shadow)
        bg.paste(sign, (x, y), sign)
        if random.random() < 0.2:  # partial occlusion (person, pillar, hand...)
            d = ImageDraw.Draw(bg)
            ox = x + random.randint(0, max(1, sign.width))
            col = tuple(random.randint(0, 200) for _ in range(3))
            d.rectangle([ox, y - 10, ox + random.randint(15, max(16, sign.width // 4)), y + sign.height + 10], fill=col)

    img = bg
    # camera effects
    if random.random() < 0.3:
        img = add_glare(img)
    if random.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 2.5)))
    if random.random() < 0.2:
        img = motion_blur(img, random.randint(3, 11))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.25))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.3))
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0, random.uniform(0, 12), arr.shape)  # sensor noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def make_sample(cls, bgs):
    if cls == "washroom_men":
        return compose(render_washroom("men"), bgs)
    if cls == "washroom_women":
        return compose(render_washroom("women"), bgs)
    if cls == "exit_sign":
        return compose(render_exit(), bgs)
    if cls in TRAFFIC_TEXT or cls == "no_entry":
        return compose(render_traffic(cls), bgs, min_scale=0.45, max_scale=0.95)
    # nothing: empty scenes + look-alike signs (hard negatives)
    r = random.random()
    sign = render_distractor() if r < 0.35 else render_road_distractor() if r < 0.6 else None
    return compose(sign, bgs)


# ---------------------------------------------------------------------------
# Printable demo props (clean, high resolution)
# ---------------------------------------------------------------------------
def make_props(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    state = random.getstate()
    random.seed(7)

    def save(img, name, bgcol=(255, 255, 255)):
        canvas = Image.new("RGB", (img.width + 80, img.height + 80), bgcol)
        canvas.paste(img, (40, 40), img)
        canvas.save(os.path.join(out_dir, name))

    # Fixed, very clear designs for the demo
    for gender, label in [("men", "GENTS"), ("women", "LADIES")]:
        W, H = 1200, 1500
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=60, fill=(0, 70, 160, 255))
        (draw_man if gender == "men" else draw_woman)(d, W / 2, H * 0.07, H * 0.62, (255, 255, 255))
        draw_text_centered(d, label, (W * 0.1, H * 0.74, W * 0.9, H * 0.93), (255, 255, 255))
        save(img, f"prop_washroom_{gender}.png")
    W, H = 1600, 640
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, W - 1, H - 1], radius=40, fill=(0, 140, 70, 255))
    draw_runner(d, W * 0.15, H * 0.12, H * 0.76, (255, 255, 255))
    draw_text_centered(d, "EXIT", (W * 0.32, H * 0.15, W * 0.78, H * 0.85), (255, 255, 255))
    draw_arrow(d, W * 0.8, H * 0.3, W * 0.15, H * 0.4, (255, 255, 255), "right")
    save(img, "prop_exit.png")
    # A text card for the Read mode demo
    W, H = 1600, 900
    card = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(card)
    draw_text_centered(d, "ROOM 204", (100, 80, W - 100, 330), (0, 0, 0))
    draw_text_centered(d, "AI & ML LAB", (100, 380, W - 100, 560), (0, 0, 0))
    draw_text_centered(d, "WASHROOM ->", (100, 640, W - 100, 820), (0, 70, 160))
    card.save(os.path.join(out_dir, "prop_read_text.png"))
    for kind in ["pedestrian_crossing", "road_work", "school_ahead", "speed_breaker", "caution", "no_entry"]:
        random.seed(11)
        S = 1000
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if kind == "no_entry":
            d.ellipse([0, 0, S - 1, S - 1], fill=(200, 25, 30)); d.rectangle([S * .16, S * .4, S * .84, S * .6], fill=(255, 255, 255))
        else:
            h = S * .87; top = (S - h) / 2; b = S * .13
            d.polygon([(S / 2, top), (S - 1, top + h), (0, top + h)], fill=(200, 25, 30))
            d.polygon([(S / 2, top + b * 2), (S - 1 - b * 1.75, top + h - b), (b * 1.75, top + h - b)], fill=(255, 255, 255))
            draw_pictogram(kind, d, (S * .3, top + h * .36, S * .7, top + h * .86), (15, 15, 15))
        save(img, f"prop_{kind}.png")
    random.setstate(state)
    print(f"Printable demo props written to {out_dir}/")


def make_preview(out_root, path, per_class=8):
    tiles = []
    for cls in CLASSES:
        folder = os.path.join(out_root, cls)
        files = sorted(os.listdir(folder))[:per_class] if os.path.isdir(folder) else []
        tiles.append([Image.open(os.path.join(folder, f)).resize((112, 112)) for f in files])
    W = 140 + per_class * 116
    grid = Image.new("RGB", (W, len(CLASSES) * 116 + 4), (255, 255, 255))
    d = ImageDraw.Draw(grid)
    f = get_font(14)
    for r, row in enumerate(tiles):
        d.text((6, r * 116 + 50), CLASSES[r], fill=(0, 0, 0), font=f)
        for c, t in enumerate(row):
            grid.paste(t, (140 + c * 116, r * 116 + 4))
    grid.save(path, quality=90)
    print(f"Preview grid for your slides: {path}")


def main():
    p = argparse.ArgumentParser(description="Drishti synthetic sign dataset generator")
    p.add_argument("--out", default="dataset", help="output folder (default: dataset)")
    p.add_argument("--per-class", type=int, default=200, help="images per class (default: 200)")
    p.add_argument("--bg-dir", default=None, help="optional folder of your own photos (corridors, classrooms) to use as backgrounds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--props-only", action="store_true", help="only generate printable demo signs")
    p.add_argument("--classes", nargs="+", default=CLASSES, choices=CLASSES, help="which classes to generate (default: all)")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if not FONTS:
        print("Note: no TrueType fonts found, using Pillow's built-in font.")

    make_props("props")
    if args.props_only:
        return

    bgs = Backgrounds(args.bg_dir)
    for cls in args.classes:
        folder = os.path.join(args.out, cls)
        os.makedirs(folder, exist_ok=True)
        for i in range(args.per_class):
            make_sample(cls, bgs).save(os.path.join(folder, f"{cls}_{i:04d}.jpg"), quality=random.randint(60, 95))
            if (i + 1) % 50 == 0 or i + 1 == args.per_class:
                sys.stdout.write(f"\r{cls:16s} {i + 1}/{args.per_class}")
                sys.stdout.flush()
        print()
    make_preview(args.out, "preview_grid.jpg")
    total = args.per_class * len(args.classes)
    print(f"\nDone: {total} images in '{args.out}/'.")
    print("Next: open https://teachablemachine.withgoogle.com/train/image and upload each folder as a class.")


if __name__ == "__main__":
    main()
