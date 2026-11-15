"""Labels file, quality check, preprocessing, clip index, M1 features, dataset inventory."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import audio as A

LABEL_COLUMNS = ["recording_id", "learner_code", "audio_file", "passage_id", "session_id",
                 "reference_tier", "labeler_id", "label_version"]


OPTIONAL_COLUMNS = ["clarity_score", "fillers_marked"]


def read_labels(cfg) -> pd.DataFrame:
    df = pd.read_csv(cfg.path("labels_csv"), dtype=str).fillna("")
    for opt in OPTIONAL_COLUMNS:  # speech-analyzer labels
        if opt not in df.columns:
            df[opt] = ""
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"labels file is missing columns: {missing}")
    if df["recording_id"].duplicated().any():
        raise ValueError("recording_id must be unique in the labels file")
    bad = set(df["reference_tier"]) - set(cfg["tiers"]) - {""}
    if bad:
        raise ValueError(f"reference_tier must be one of {cfg['tiers']} or empty; found {bad}")
    if (df["learner_code"] == "").any():
        raise ValueError("every row needs an anonymized learner_code")
    versions = set(df.loc[df["reference_tier"] != "", "label_version"])
    if len(versions) > 1:
        raise ValueError(f"mixed label versions in one file: {versions}. Train one frozen version only.")
    if cfg["mode"] == "study" and versions and versions != {str(cfg["gate"]["label_file_version"])}:
        raise ValueError("label_version in labels.csv does not match gate.label_file_version")
    return df


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["reference_tier"] != ""].copy()


def work(cfg, *parts) -> Path:
    p = cfg.path("work_dir").joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run_qc_and_preprocess(cfg) -> pd.DataFrame:
    """Quality-check every recording; preprocess and cache those that pass."""
    df = read_labels(cfg)
    ac, sr = cfg["audio"], cfg["audio"]["sample_rate"]
    rows = []
    for r in df.itertuples():
        y = A.load_audio(cfg.path("raw_audio_dir") / r.audio_file, sr)
        qc = A.quality_check(y, sr, ac["qc"], ac["trim_top_db"])
        if qc.passed:
            np.save(work(cfg, "processed", f"{r.recording_id}.npy"), A.preprocess(y, ac))
        rows.append({"recording_id": r.recording_id, "learner_code": r.learner_code,
                     "quality_status": "passed" if qc.passed else "failed", **qc.as_dict(),
                     "trim_start_s": round(A.trim_start_s(y, ac["trim_top_db"], sr), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(work(cfg, "qc.csv"), index=False)
    print(f"Quality check: {(out.quality_status == 'passed').sum()} passed, "
          f"{(out.quality_status == 'failed').sum()} failed -> {work(cfg, 'qc.csv')}")
    return out


def load_processed(cfg, recording_id: str) -> np.ndarray:
    return np.load(work(cfg, "processed", f"{recording_id}.npy"))


def usable_recordings(cfg) -> pd.DataFrame:
    """Labeled recordings that passed the quality check."""
    df = labeled(read_labels(cfg))
    qc = pd.read_csv(work(cfg, "qc.csv"), dtype={"recording_id": str})
    ok = set(qc.loc[qc.quality_status == "passed", "recording_id"])
    return df[df.recording_id.isin(ok)].reset_index(drop=True)


def manual_exclusions(cfg) -> set[tuple[str, int]]:
    p = cfg.root / "data" / "excluded_clips.csv"
    if not p.exists():
        return set()
    ex = pd.read_csv(p, dtype={"recording_id": str})
    return {(r.recording_id, int(r.clip_index)) for r in ex.itertuples()}


def clips_for(cfg, recording_id: str, clip_s: float, excluded=frozenset()):
    """Kept clips of one recording: list of (clip_index, waveform)."""
    ac = cfg["audio"]
    y = load_processed(cfg, recording_id)
    segs = A.segment(y, ac["sample_rate"], clip_s, ac["clip_overlap"], ac["trim_top_db"],
                     ac["clip_min_active_fraction"])
    kept = [(i, c) for i, _, c, _, k in segs if k and (recording_id, i) not in excluded]
    if not kept:  # never drop a whole recording silently: keep its most active clip
        i, _, c, _, _ = max(segs, key=lambda s: s[3])
        kept = [(i, c)]
    return kept, segs


def clip_index(cfg) -> pd.DataFrame:
    """Log of clips per recording and clip length, including how many were dropped (Table 3)."""
    rec = usable_recordings(cfg)
    excluded = manual_exclusions(cfg)
    rows = []
    for L in cfg["audio"]["clip_lengths_s"]:
        for r in rec.itertuples():
            kept, segs = clips_for(cfg, r.recording_id, L, excluded)
            rows.append({"clip_s": L, "recording_id": r.recording_id, "learner_code": r.learner_code,
                         "clips_total": len(segs), "clips_kept": len(kept),
                         "clips_dropped": len(segs) - len(kept)})
    out = pd.DataFrame(rows)
    out.to_csv(work(cfg, "clip_log.csv"), index=False)
    return out


def handcrafted_table(cfg) -> pd.DataFrame:
    rec = usable_recordings(cfg)
    sr, top_db = cfg["audio"]["sample_rate"], cfg["audio"]["trim_top_db"]
    feats = [A.handcrafted_features(load_processed(cfg, rid), sr, top_db) for rid in rec.recording_id]
    out = pd.DataFrame(feats, columns=A.HANDCRAFTED_NAMES)
    out.insert(0, "recording_id", rec.recording_id.values)
    out.to_csv(work(cfg, "handcrafted.csv"), index=False)
    return out


def inventory(cfg) -> pd.DataFrame:
    """Training Plan Table 2 (dataset inventory)."""
    rec = usable_recordings(cfg)
    qc = pd.read_csv(work(cfg, "qc.csv"), dtype={"recording_id": str}).set_index("recording_id")
    rec["minutes"] = rec.recording_id.map(qc["duration_s"]) / 60
    tiers = cfg["tiers"]
    table = {}
    table["Learners"] = rec.groupby("reference_tier").learner_code.nunique()
    table["Recordings (passed quality check)"] = rec.groupby("reference_tier").size()
    table["Total audio (minutes)"] = rec.groupby("reference_tier").minutes.sum().round(1)
    clog = work(cfg, "clip_log.csv")
    if clog.exists():
        cl = pd.read_csv(clog, dtype={"recording_id": str}).merge(rec[["recording_id", "reference_tier"]])
        for L, g in cl.groupby("clip_s"):
            table[f"Clips after segmentation ({L} s)"] = g.groupby("reference_tier").clips_kept.sum()
    test_p = work(cfg, "splits", "test_learners.csv")
    if test_p.exists():
        t = pd.read_csv(test_p, dtype=str)
        table["Learners in locked test set"] = t.groupby("reference_tier").learner_code.nunique()
    out = pd.DataFrame(table).T.reindex(columns=tiers).fillna(0)
    out["Total"] = out.sum(axis=1)
    return out.map(lambda v: f"{v:g}")


# ---------------------------------------------------------------------------
# Speech-analyzer labels: filler marks, clarity scores, hand-timed checks
# ---------------------------------------------------------------------------
def filler_marks(cfg) -> dict:
    """recording_id -> list of (start_s, end_s) in PROCESSED-audio time, for every recording with
    fillers_marked == yes (an empty list means it was checked and has no fillers).
    Marks are made on the original file; the trimmed start is subtracted here."""
    df = read_labels(cfg)
    if "fillers_marked" not in df.columns:
        raise ValueError("labels.csv needs a 'fillers_marked' column (yes/no) for the speech analyzer")
    marked = set(df.loc[df.fillers_marked.str.lower().isin(["yes", "y", "true", "1"]), "recording_id"])
    p = cfg.root / "data" / "filler_marks.csv"
    fm = pd.read_csv(p, dtype={"recording_id": str}) if p.exists() else pd.DataFrame(
        columns=["recording_id", "start_s", "end_s", "label"])
    unknown = set(fm.recording_id) - marked
    if unknown:
        raise ValueError(f"filler_marks.csv has marks for recordings not flagged fillers_marked=yes: {sorted(unknown)[:5]}")
    qc = pd.read_csv(work(cfg, "qc.csv"), dtype={"recording_id": str}).set_index("recording_id")
    out = {}
    for rid in marked:
        if rid not in qc.index or qc.loc[rid, "quality_status"] != "passed":
            continue
        off, dur = qc.loc[rid, "trim_start_s"], qc.loc[rid, "duration_s"]
        g = fm[fm.recording_id == rid]
        out[rid] = [(max(0.0, a - off), min(dur, b - off)) for a, b in zip(g.start_s, g.end_s) if b - off > 0]
    return out


def import_audacity_labels(cfg, folder) -> pd.DataFrame:
    """Convert Audacity label exports (<recording_id>.txt: start<TAB>end<TAB>label) into
    data/filler_marks.csv. Only labels named in filler.label_names are kept."""
    names = {n.lower() for n in cfg["filler"]["label_names"]}
    rows = []
    for f in sorted(Path(folder).glob("*.txt")):
        for line in f.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[2].strip().lower() in names:
                rows.append({"recording_id": f.stem, "start_s": float(parts[0]), "end_s": float(parts[1]),
                             "label": parts[2].strip()})
    out = pd.DataFrame(rows, columns=["recording_id", "start_s", "end_s", "label"])
    out.to_csv(cfg.root / "data" / "filler_marks.csv", index=False)
    print(f"Imported {len(out)} filler marks from {folder} -> data/filler_marks.csv. "
          "Set fillers_marked=yes in labels.csv for every recording that was checked (including ones with none).")
    return out
