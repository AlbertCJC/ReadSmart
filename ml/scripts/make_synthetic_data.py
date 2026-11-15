"""Generate SYNTHETIC "recordings" to test the pipeline end to end.

These are buzzy tone sequences, not speech, and not children. The tier labels are
arbitrary: "High" files simply get more pauses and more noise so the pipeline has
something to find. Nothing learned from them means anything about real learners,
and they must never be mixed into the study dataset.

Usage:  python scripts/make_synthetic_data.py --out synthetic [--learners 36] [--shuffle-labels]
Writes: synthetic/data/raw/*.wav, synthetic/data/labels.csv, synthetic/data/double_labels.csv,
        synthetic/config.yaml (practice mode, paths inside synthetic/)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy.signal import butter, lfilter

SR = 16000
TIERS = ["Low", "Moderate", "High"]
# arbitrary generator settings per tier (pause probability, noise level, syllable-length jitter)
STYLE = {"Low": (0.08, 0.02, 0.15), "Moderate": (0.16, 0.06, 0.30), "High": (0.26, 0.12, 0.45)}


def syllable(rng, f0, dur, noise):
    t = np.arange(int(dur * SR)) / SR
    f = f0 * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(2, 5) * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    tone = sum(np.sin(k * ph) / k for k in range(1, 12))
    lo = rng.uniform(300, 900)
    b, a = butter(2, [lo / (SR / 2), min(lo * 4, 7000) / (SR / 2)], btype="band")
    x = lfilter(b, a, tone + noise * 8 * rng.standard_normal(len(t)))
    env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 0.6
    return (x * env).astype(np.float32)


def filler(rng, f0, dur):
    """A flat, sustained vowel-like tone standing in for "uh"/"um" (no vibrato, low band)."""
    t = np.arange(int(dur * SR)) / SR
    ph = 2 * np.pi * f0 * t
    tone = sum(np.sin(k * ph) / k ** 1.5 for k in range(1, 8))
    b, a = butter(2, [350 / (SR / 2), 1500 / (SR / 2)], btype="band")
    x = lfilter(b, a, tone)
    env = np.minimum(1, np.minimum(t / 0.05, (dur - t) / 0.08))
    return (0.8 * x * env).astype(np.float32)


# chance per syllable slot of inserting a filler (arbitrary)
FILLER_P = {"Low": 0.01, "Moderate": 0.03, "High": 0.06}


def reading(rng, tier, seconds):
    """Returns audio and the generator's own truth: filler spans, pauses (>= 0.25 s), syllable count."""
    p_pause, noise, jitter = STYLE[tier]
    f0 = rng.uniform(220, 300)
    out = [np.zeros(int(rng.uniform(0.3, 0.8) * SR), np.float32)]  # leading silence
    t = len(out[0]) / SR
    total, fillers, gaps, n_syl = 0.0, [], [], 0
    while total < seconds:
        if rng.random() < FILLER_P[tier]:
            d = rng.uniform(0.35, 0.7)
            seg = filler(rng, f0 * 0.8, d)
            fillers.append((round(t, 3), round(t + len(seg) / SR, 3)))
        else:
            d = 0.22 * (1 + rng.uniform(-jitter, jitter))
            seg = syllable(rng, f0 * rng.uniform(0.9, 1.1), d, noise)
            n_syl += 1
        out.append(seg)
        t += len(seg) / SR
        gap = rng.uniform(0.4, 1.2) if rng.random() < p_pause else rng.uniform(0.02, 0.08)
        g = np.zeros(int(gap * SR), np.float32)
        out.append(g)
        gaps.append((t, t + len(g) / SR))
        t += len(g) / SR
        total += d + gap
    gaps = gaps[:-1]  # last gap runs into trailing silence
    out.append(np.zeros(int(rng.uniform(0.3, 0.8) * SR), np.float32))
    y = np.concatenate(out)
    y += 0.002 * rng.standard_normal(len(y)).astype(np.float32)  # room hiss
    pauses = [b - a for a, b in gaps if b - a >= 0.25]
    truth = {"fillers": fillers, "pause_count": len(pauses), "total_pause_s": round(sum(pauses), 2),
             "syllable_count": n_syl}
    return 0.3 * y / (np.abs(y).max() + 1e-9), truth


CLARITY = {"Low": (4, 5), "Moderate": (3, 4), "High": (1, 3)}  # arbitrary 1-5 fake rater scores


SYNTHETIC_CRITERIA = """# MADE-UP RULES FOR THE SYNTHETIC TEST ONLY. These numbers are invented to exercise the tier step.
# They are NOT tier criteria and must never be used or cited. Real criteria go in criteria/tier_criteria.yaml.
version: "synthetic-test-only"
approved_by: "nobody (synthetic test)"
default_tier: Low
rules:
  - tier: High
    any:
      - {metric: clarity_score, op: "<=", value: 2.5}
      - {metric: fillers_per_min, op: ">=", value: 10}
  - tier: Moderate
    any:
      - {metric: clarity_score, op: "<=", value: 3.6}
      - {metric: fillers_per_min, op: ">=", value: 5}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--learners", type=int, default=36)
    ap.add_argument("--seconds", type=float, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shuffle-labels", action="store_true", help="sanity check: scores should fall to chance")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # imbalanced like the plan expects: Low most common, High least
    n_high, n_mod = max(6, a.learners // 5), max(6, a.learners // 3)
    tiers = ["High"] * n_high + ["Moderate"] * n_mod + ["Low"] * (a.learners - n_high - n_mod)
    rows, rid, marks, timing = [], 1, [], []
    for i, tier in enumerate(tiers):
        code = f"SYN{i + 1:03d}"
        if i in (0, 5):  # two learners get a failed first attempt (too quiet / too short), then a retake
            bad = reading(rng, tier, 4.0)[0] if i == 0 else reading(rng, tier, a.seconds)[0] * 0.001
            f = f"{code}_take1.wav"
            sf.write(raw / f, bad, SR, subtype="PCM_16")
            rows.append([f"R{rid:04d}", code, f, "P-SYN", "S1", "", "", "", "", "no"])
            rid += 1
        f = f"{code}_take{2 if i in (0, 5) else 1}.wav"
        y, truth = reading(rng, tier, a.seconds * rng.uniform(0.8, 1.2))
        sf.write(raw / f, y, SR, subtype="PCM_16")
        r = f"R{rid:04d}"
        marked = i % 3 != 2  # fillers marked in about two thirds of recordings, like a labeled subset
        clarity = int(rng.integers(CLARITY[tier][0], CLARITY[tier][1] + 1))
        rows.append([r, code, f, "P-SYN", "S1" if i % 2 else "S2", tier, "rater_A", "synthetic_v0",
                     clarity, "yes" if marked else "no"])
        if marked:
            marks += [{"recording_id": r, "start_s": a_, "end_s": b_, "label": "uh"} for a_, b_ in truth["fillers"]]
        if i % 2 == 0:  # hand-timed check on half the recordings
            timing.append({"recording_id": r, **{k: truth[k] for k in ("pause_count", "total_pause_s", "syllable_count")}})
        rid += 1
    df = pd.DataFrame(rows, columns=["recording_id", "learner_code", "audio_file", "passage_id", "session_id",
                                     "reference_tier", "labeler_id", "label_version", "clarity_score",
                                     "fillers_marked"])
    df["notes"] = "SYNTHETIC - not a learner"
    if a.shuffle_labels:
        m = df.reference_tier != ""
        df.loc[m, "reference_tier"] = rng.permutation(df.loc[m, "reference_tier"].to_numpy())
    df.to_csv(root / "data" / "labels.csv", index=False)
    pd.DataFrame(marks, columns=["recording_id", "start_s", "end_s", "label"]).to_csv(
        root / "data" / "filler_marks.csv", index=False)
    pd.DataFrame(timing).to_csv(root / "data" / "timing_checks.csv", index=False)
    (root / "criteria_synthetic.yaml").write_text(SYNTHETIC_CRITERIA)
    lab = df[df.reference_tier != ""].sample(12, random_state=a.seed)
    flip = {"Low": "Moderate", "Moderate": "Low", "High": "Moderate"}
    b = [flip[t] if rng.random() < 0.2 else t for t in lab.reference_tier]
    pd.DataFrame({"recording_id": lab.recording_id, "rater_a_tier": lab.reference_tier, "rater_b_tier": b}
                 ).to_csv(root / "data" / "double_labels.csv", index=False)
    here = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((here / "config.yaml").read_text())
    cfg["mode"] = "practice"
    # quick CPU check: one small M5 setting on 10 s clips only
    cfg["paths"]["criteria_file"] = "criteria_synthetic.yaml"
    cfg["clarity"].update({"scale_min": 1, "scale_max": 5})
    cfg["models"]["M5"].update({"lr": [0.001], "dropout": [0.3], "max_epochs": 15, "patience": 5, "clip_lengths_s": [10]})
    (root / "config.yaml").write_text(
        "# AUTO-GENERATED for synthetic practice runs. Numbers are NOT results.\n" + yaml.safe_dump(cfg, sort_keys=False))
    print(f"Wrote {len(df)} synthetic files for {len(tiers)} fake learners to {root} "
          f"({pd.Series(tiers).value_counts().to_dict()})")


if __name__ == "__main__":
    main()
