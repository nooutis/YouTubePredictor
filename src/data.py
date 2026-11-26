import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DAY_COLS = [f"day{i}" for i in range(20)]
MAX_REL_DROP = 0.20   # drop a video if its cumulative views fall by more than 20% from the running maximum
N_FOLDS = 10          # fold 0 = test (10%), fold 1 = val (10%), the rest = train (80%)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]")


def clean(df, log):
    log.append(f"initial rows: {len(df)}")

    # 1. Cumulative views must not decrease: fix with a running max, drop videos with a large drop.
    d = df[DAY_COLS].to_numpy(dtype=float)
    running_max = np.maximum.accumulate(d, axis=1)
    max_drop = (running_max - d).max(axis=1)
    rel_drop = max_drop / np.maximum(running_max[:, -1], 1)
    log.append(f"non-monotonic rows (fixed with running max): {(max_drop > 0).sum()}")
    keep = rel_drop <= MAX_REL_DROP
    log.append(f"dropped for a drop > {MAX_REL_DROP:.0%}: {(~keep).sum()}")
    df = df[keep].copy()
    df[DAY_COLS] = np.maximum.accumulate(df[DAY_COLS].to_numpy(dtype=float), axis=1)

    # 2. day19 == 0: the views-to-subscribers ratio has no logarithm.
    keep = df["day19"] > 0
    log.append(f"dropped for day19 == 0: {(~keep).sum()}")
    df = df[keep].copy()

    # 3. Flag truncated titles (kept, not dropped).
    df["title_truncated"] = df["title"].str.endswith("…").astype(int)
    return df


def add_target(df):
    # Log of views relative to channel size: removes most of the "big channel = many views" effect.
    df["y"] = np.log(df["day19"] / df["subscribers"])
    return df


def make_split(df, seed):
    # Stratify by category, group by channel so no channel appears in more than one split.
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, idx) in enumerate(sgkf.split(df, df["category"], groups=df["channel_title"])):
        df.iloc[idx, df.columns.get_loc("fold")] = fold
    df["split"] = np.where(df.fold == 0, "test", np.where(df.fold == 1, "val", "train"))

    # The "top quartile" threshold is computed on the train split only,
    # so the test set does not influence the label definition.
    thr = df.loc[df.split == "train", "y"].quantile(0.75)
    df["y_top"] = (df["y"] >= thr).astype(int)
    return df, thr


def make_features(df):
    t = df["title"]
    f = pd.DataFrame({"video_id": df["video_id"]})
    f["log_subscribers"] = np.log(df["subscribers"])
    f["log_duration"] = np.log(df["duration"])
    f["title_len"] = t.str.len()
    f["title_n_words"] = t.str.split().str.len()
    f["title_upper_ratio"] = t.apply(lambda s: sum(c.isupper() for c in s) / max(sum(c.isalpha() for c in s), 1))
    f["title_n_caps_words"] = t.apply(lambda s: sum(w.isupper() and len(w) > 1 for w in s.split()))
    f["title_n_digits"] = t.str.count(r"\d")
    f["title_has_question"] = t.str.contains(r"\?").astype(int)
    f["title_has_exclaim"] = t.str.contains("!").astype(int)
    f["title_has_emoji"] = t.apply(lambda s: int(bool(EMOJI_RE.search(s))))
    f["title_has_brackets"] = t.str.contains(r"[\(\[\|]").astype(int)
    f["title_truncated"] = df["title_truncated"].values
    cats = pd.get_dummies(df["category"], prefix="cat").astype(int)
    return pd.concat([f.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)


def checks(df, thr):
    lines = [f"\ny_top threshold (75th percentile on train): {thr:.3f}"]
    lines.append("\nsplit sizes:\n" + df["split"].value_counts().to_string())
    lines.append("\ny per split:\n" + df.groupby("split")["y"].describe()[["count", "mean", "std"]].round(3).to_string())
    lines.append("\nshare of y_top per split:\n" + df.groupby("split")["y_top"].mean().round(3).to_string())
    ch = df.groupby("channel_title")["split"].nunique()
    lines.append(f"\nchannels appearing in more than one split: {(ch > 1).sum()} (must be 0)")
    cat = pd.crosstab(df["category"], df["split"], normalize="columns").round(3)
    lines.append("\ncategory share per split:\n" + cat.to_string())
    r = np.corrcoef(df["y"], df["log_subscribers"])[0, 1]
    r_raw = np.corrcoef(np.log(df["day19"]), df["log_subscribers"])[0, 1]
    lines.append(f"\ncorrelation of log(subscribers) with log(day19): {r_raw:.3f}; with y: {r:.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--valid", default=None, help="valid_images.csv (video_id, valid_img)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv)
    log = []

    df = clean(df, log)

    if a.valid:
        v = pd.read_csv(a.valid)
        good = set(v.loc[v.valid_img, "video_id"])
        keep = df["video_id"].isin(good)
        log.append(f"dropped for missing/invalid thumbnail: {(~keep).sum()}")
        df = df[keep].copy()

    df = add_target(df).reset_index(drop=True)
    df, thr = make_split(df, a.seed)
    feats = make_features(df)
    log.append(f"final: {len(df)} videos, {df['channel_title'].nunique()} channels")

    df[["video_id", "channel_title", "category", "y", "y_top", "fold", "split"]].to_csv(out / "split.csv", index=False)
    feats.to_csv(out / "features.csv", index=False)

    report = "\n".join(log) + "\n" + checks(df.merge(feats[["video_id", "log_subscribers"]], on="video_id"), thr)
    (out / "report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()