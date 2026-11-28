"""Interpretable thumbnail features: faces + emotion, text (OCR), and simple colour/contrast statistics.
 
Works locally and on Kaggle. Resumable: rows already present in --out are skipped.
 
Examples:
    # quick test on 50 images
    python src/image_features.py --split data/split.csv --images thumbs/ --out data/image_features.csv --limit 50
 
    # full run on Kaggle (GPU), with titles to compute the text/title overlap feature
    python src/image_features.py --split data/split.csv \
        --images /kaggle/input/datasets/charlottecarretta/youtube-datased-cleaned-with-images/thumbnails \
        --meta /kaggle/input/datasets/charlottecarretta/youtube-datased-cleaned-with-images/youtube_dataset_local_images.csv \
        --out image_features.csv
 
Install (Kaggle needs Internet ON):  pip install deepface easyocr
Use --no-faces / --no-ocr to skip a part (the colour/contrast features need only OpenCV).
 
Output: image_features.csv, one row per video_id (same key as split.csv / features.csv).
Videos where no face was found get n_faces = 0 and emotion columns = 0.
"""
import argparse
import re
from pathlib import Path
 
import cv2
import numpy as np
import pandas as pd
 
W, H = 320, 180
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
 
 
# ---------- classical features (OpenCV only) ----------
def colour_features(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = [img_bgr[..., i].astype(float) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    edges = cv2.Canny(gray, 100, 200)
    return {
        "brightness": gray.mean() / 255,
        "contrast": gray.std() / 255,
        "saturation": hsv[..., 1].mean() / 255,
        "colorfulness": colorfulness / 255,
        "edge_density": (edges > 0).mean(),
    }
 
 
# ---------- faces + emotion (DeepFace) ----------
def face_features(path, detector):
    from deepface import DeepFace
 
    out = {"n_faces": 0, "face_area": 0.0, "face_cx": 0.0, "face_cy": 0.0}
    out.update({f"emo_{e}": 0.0 for e in EMOTIONS})
    try:
        res = DeepFace.analyze(str(path), actions=["emotion"], detector_backend=detector,
                               enforce_detection=False, silent=True)
    except Exception:
        return out
    if isinstance(res, dict):
        res = [res]
    # with enforce_detection=False DeepFace may return the whole image as a "face": filter those out
    faces = [f for f in res
             if f.get("face_confidence", 1) > 0.8
             and f["region"]["w"] * f["region"]["h"] < 0.95 * W * H]
    out["n_faces"] = len(faces)
    if faces:
        main = max(faces, key=lambda f: f["region"]["w"] * f["region"]["h"])
        rg = main["region"]
        out["face_area"] = rg["w"] * rg["h"] / (W * H)
        out["face_cx"] = (rg["x"] + rg["w"] / 2) / W
        out["face_cy"] = (rg["y"] + rg["h"] / 2) / H
        for e in EMOTIONS:
            out[f"emo_{e}"] = main["emotion"].get(e, 0.0) / 100
    return out
 
 
# ---------- text (EasyOCR) ----------
def text_features(path, reader, title, min_conf=0.4):
    res = [(b, t, c) for b, t, c in reader.readtext(str(path)) if c >= min_conf]
    words = [w for _, t, _ in res for w in t.split()]
    text = " ".join(words)
    area = sum(abs(b[2][0] - b[0][0]) * abs(b[2][1] - b[0][1]) for b, _, _ in res)
    heights = [abs(b[2][1] - b[0][1]) for b, _, _ in res]
    letters = [c for c in text if c.isalpha()]
    out = {
        "has_text": int(len(words) > 0),
        "n_text_boxes": len(res),
        "n_text_words": len(words),
        "text_area": min(area / (W * H), 1.0),   # sum of box areas; boxes may overlap slightly
        "text_height": float(np.mean(heights)) / H if heights else 0.0,
        "text_upper_ratio": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        "text_has_digit": int(any(c.isdigit() for c in text)),
        "text_has_exclaim_or_question": int(bool(re.search(r"[!?]", text))),
    }
    if title is not None:
        tw = set(re.findall(r"\w+", title.lower()))
        ow = set(re.findall(r"\w+", text.lower()))
        out["text_title_overlap"] = len(tw & ow) / len(ow) if ow else 0.0
    return out
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="split.csv (video_id list)")
    ap.add_argument("--images", required=True, help="folder with <video_id>.jpg")
    ap.add_argument("--out", default="image_features.csv")
    ap.add_argument("--meta", default=None, help="CSV with video_id,title (enables text_title_overlap)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N videos (test run)")
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--detector", default="opencv", help="DeepFace detector_backend (opencv is fast; try retinaface/mtcnn if faces are missed)")
    ap.add_argument("--no-faces", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()
 
    ids = pd.read_csv(a.split, usecols=["video_id"]).video_id.tolist()
    if a.limit:
        ids = ids[: a.limit]
    titles = {}
    if a.meta:
        m = pd.read_csv(a.meta, usecols=["video_id", "title"])
        titles = dict(zip(m.video_id, m.title))
 
    out_path = Path(a.out)
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    done_ids = set(done.video_id) if len(done) else set()
    todo = [v for v in ids if v not in done_ids]
    print(f"{len(ids)} videos, {len(done_ids)} already done, {len(todo)} to process")
 
    reader = None
    if not a.no_ocr:
        import easyocr
        try:
            import torch
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False
        reader = easyocr.Reader(["en"], gpu=gpu)
 
    rows, parts = [], [done] if len(done) else []
 
    def flush():
        nonlocal rows
        if rows:
            parts.append(pd.DataFrame(rows))
            pd.concat(parts, ignore_index=True).to_csv(out_path, index=False)
            rows = []
 
    for i, vid in enumerate(todo, 1):
        path = Path(a.images) / f"{vid}.jpg"
        row = {"video_id": vid}
        img = cv2.imread(str(path))
        if img is None:
            row["read_ok"] = 0
        else:
            row["read_ok"] = 1
            row.update(colour_features(img))
            if not a.no_faces:
                row.update(face_features(path, a.detector))
            if reader is not None:
                row.update(text_features(path, reader, titles.get(vid)))
        rows.append(row)
        if i % a.checkpoint_every == 0:
            flush()
            print(f"  {i}/{len(todo)}")
    flush()
    print(f"saved to {out_path}")
 
 
if __name__ == "__main__":
    main()
 