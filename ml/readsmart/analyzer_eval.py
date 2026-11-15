"""Evaluation of the speech analyzer and the tier step.

  timing-check     automatic pauses / syllable peaks vs hand-timed values (development learners only)
  analyzer-cv      learner-grouped CV of the filler and clarity heads (+ tier if criteria are set)
  analyzer-final   train on all development learners, evaluate ONCE on the locked test set, save the model
  analyze          one WAV -> speech metrics + tier
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import splits as S
from .analyzer import METRIC_NAMES, SpeechAnalyzer, frame_labels, load_frames
from .data import filler_marks, load_processed, usable_recordings, work
from .metrics import evaluate, plot_confusion, report_table
from .runlog import append_run_log, new_run_dir
from .tiers import assign_tier, load_criteria


def criteria(cfg):
    p = Path(cfg["paths"]["criteria_file"])
    return load_criteria(p if p.is_absolute() else cfg.root / p, cfg["tiers"])


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def match_events(pred, true, min_overlap: float = 0.5):
    """Greedy one-to-one matching: a predicted event matches a marked filler if their overlap covers at
    least min_overlap of the shorter of the two. Returns (tp, fp, fn)."""
    used, tp = set(), 0
    for a, b in pred:
        for j, (c, d) in enumerate(true):
            if j in used:
                continue
            ov = min(b, d) - max(a, c)
            if ov > 0 and ov >= min_overlap * min(b - a, d - c):
                used.add(j)
                tp += 1
                break
    return tp, len(pred) - tp, len(true) - tp


def filler_scores(rows: list[dict]) -> dict:
    """rows: dicts with pred_events, true_events, frame_pred, frame_true."""
    if not rows:
        return {}
    tp = fp = fn = 0
    ftp = ffp = ffn = 0
    count_err = []
    for r in rows:
        a, b, c = match_events(r["pred_events"], r["true_events"])
        tp, fp, fn = tp + a, fp + b, fn + c
        fpred, ftrue = r["frame_pred"], r["frame_true"]
        n = min(len(fpred), len(ftrue))
        fpred, ftrue = fpred[:n].astype(bool), ftrue[:n].astype(bool)
        ftp += int((fpred & ftrue).sum())
        ffp += int((fpred & ~ftrue).sum())
        ffn += int((~fpred & ftrue).sum())
        count_err.append(len(r["pred_events"]) - len(r["true_events"]))
    prf = lambda t, p, n: (t / (t + p) if t + p else 0.0, t / (t + n) if t + n else 0.0)
    p, rc = prf(tp, fp, fn)
    fp_, fr_ = prf(ftp, ffp, ffn)
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    return {"filler_recordings": len(rows), "filler_marked_events": tp + fn,
            "filler_event_precision": p, "filler_event_recall": rc, "filler_event_f1": f1(p, rc),
            "filler_frame_precision": fp_, "filler_frame_recall": fr_, "filler_frame_f1": f1(fp_, fr_),
            "filler_count_mae": float(np.mean(np.abs(count_err))),
            "filler_count_bias": float(np.mean(count_err))}


def agreement(pred, true, prefix: str) -> dict:
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    out = {f"{prefix}_n": len(true)}
    if len(true) == 0:
        return out
    out[f"{prefix}_mae"] = float(np.mean(np.abs(pred - true)))
    out[f"{prefix}_bias"] = float(np.mean(pred - true))
    if len(true) >= 3 and np.std(true) > 0 and np.std(pred) > 0:
        out[f"{prefix}_pearson_r"] = float(pearsonr(pred, true)[0])
        out[f"{prefix}_spearman_rho"] = float(spearmanr(pred, true)[0])
    return out


def _evaluate_recordings(d_rec, analyzer, marks, rids, crit, tiers):
    """Analyze recordings; return per-recording table and score dict."""
    rows, frows = [], []
    rated = d_rec.set_index("recording_id").clarity_score.astype(str).str.strip()
    for rid in rids:
        m, ev = analyzer.analyze_recording(rid)
        t = assign_tier(m, crit)
        rows.append({"recording_id": rid, **m, "tier": t["tier"], "tier_status": t["status"],
                     "reference_tier": d_rec.set_index("recording_id").reference_tier[rid],
                     "rated_clarity": rated[rid], "filler_events": json.dumps(ev)})
        if rid in marks:
            fr = load_frames(analyzer.cfg, rid)
            probs = analyzer.filler.frame_probs(fr)
            frows.append({"pred_events": ev, "true_events": marks[rid],
                          "frame_pred": probs >= analyzer.filler.c["threshold"],
                          "frame_true": frame_labels(len(fr), marks[rid])})
    tab = pd.DataFrame(rows)
    scores = filler_scores(frows)
    r = tab[tab.rated_clarity != ""]
    scores.update(agreement(r.clarity_score, r.rated_clarity.astype(float), "clarity"))
    if crit is not None:
        tidx = {t: i for i, t in enumerate(tiers)}
        m = evaluate(tab.reference_tier.map(tidx), tab.tier.map(tidx), tiers)
        scores.update({f"tier_{k}": v for k, v in m.items() if k != "confusion_matrix"})
        scores["tier_confusion_matrix"] = json.dumps(m["confusion_matrix"])
    return tab, scores


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def timing_check(cfg) -> pd.DataFrame:
    """Compare automatic timing measures with hand-timed values in data/timing_checks.csv.
    Only development learners are used, so tuning silence_db / min_pause_s never touches the test set."""
    from .timing import timing_measures
    p = cfg.root / "data" / "timing_checks.csv"
    hand = pd.read_csv(p, dtype={"recording_id": str})
    test = set(pd.read_csv(work(cfg, "splits", "test_learners.csv"), dtype=str).learner_code)
    rec = usable_recordings(cfg)
    dev = set(rec.loc[~rec.learner_code.isin(test), "recording_id"])
    hand = hand[hand.recording_id.isin(dev)]
    sr = cfg["audio"]["sample_rate"]
    auto = []
    for rid in hand.recording_id:
        auto.append({"recording_id": rid, **timing_measures(load_processed(cfg, rid), sr, cfg["timing"])})
    auto = pd.DataFrame(auto).set_index("recording_id")
    out = {}
    for col in ("pause_count", "total_pause_s", "syllable_count"):
        if col in hand.columns:
            h = hand.set_index("recording_id")[col].dropna()
            out.update(agreement(auto.loc[h.index, col], h, col))
    print(f"Timing check on {len(hand)} hand-timed development recordings:\n" + json.dumps(out, indent=2))
    return pd.DataFrame([out])


def analyzer_cv(cfg) -> Path:
    rec = usable_recordings(cfg)
    marks = filler_marks(cfg)
    _, folds = S.load_splits(cfg)
    test_l = S.load_splits(cfg)[0].learner_code.to_numpy()
    crit = criteria(cfg)
    run = new_run_dir(cfg, "analyzer_cv")
    fold_rows, preds = [], []
    for f in sorted(folds.fold.unique()):
        tr, va = S.fold_learners(folds, f)
        S.check_no_leakage(tr, va, test_l)
        an = SpeechAnalyzer(cfg).fit(rec, marks, tr)
        rids = rec.loc[rec.learner_code.isin(set(va)), "recording_id"].tolist()
        tab, sc = _evaluate_recordings(rec, an, marks, rids, crit, cfg["tiers"])
        fold_rows.append({"fold": f, **an.n_train, **sc})
        preds.append(tab.assign(fold=f))
        print(f"  fold {f}: filler event F1 {sc.get('filler_event_f1', float('nan')):.3f}  "
              f"clarity MAE {sc.get('clarity_mae', float('nan')):.3f}" +
              (f"  tier macro F1 {sc['tier_f1_macro']:.3f}" if crit else "  tier: criteria not set"))
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(run / "analyzer_cv_folds.csv", index=False)
    pd.concat(preds).to_csv(run / "analyzer_cv_predictions.csv", index=False)
    num = folds_df.select_dtypes("number").drop(columns=["fold"])
    summary = pd.DataFrame({"mean": num.mean(), "sd": num.std()})
    summary.to_csv(run / "analyzer_cv_summary.csv")
    print(summary.round(3).to_string())
    append_run_log(cfg, run, "analyzer_cv", {"criteria_version": crit["version"] if crit else None,
                                             "filler_event_f1": float(num.get("filler_event_f1", pd.Series([np.nan])).mean())})
    print(f"Analyzer CV -> {run}")
    return run


def analyzer_final(cfg, version_label: str, force_reason: str | None = None) -> Path:
    lock = work(cfg, "splits", "ANALYZER_TEST_SET_OPENED.json")
    if lock.exists() and not force_reason:
        raise RuntimeError(f"The locked test set was already used for the analyzer ({lock}). "
                           "Re-running needs --force-reason, which is written to the run log.")
    rec = usable_recordings(cfg)
    marks = filler_marks(cfg)
    test, folds = S.load_splits(cfg)
    dev, test_l = S.all_dev_learners(folds), test.learner_code.to_numpy()
    S.check_no_leakage(dev, test_l, test_l, eval_is_test=True)
    crit = criteria(cfg)
    run = new_run_dir(cfg, "analyzer_final_test")
    an = SpeechAnalyzer(cfg).fit(rec, marks, dev)
    rids = rec.loc[rec.learner_code.isin(set(test_l)), "recording_id"].tolist()
    tab, sc = _evaluate_recordings(rec, an, marks, rids, crit, cfg["tiers"])
    tab.to_csv(run / "test_predictions.csv", index=False)
    (run / "test_scores.json").write_text(json.dumps(sc, indent=2, default=float))
    if crit is not None:
        tidx = {t: i for i, t in enumerate(cfg["tiers"])}
        m = evaluate(tab.reference_tier.map(tidx), tab.tier.map(tidx), cfg["tiers"])
        report_table(m, cfg["tiers"]).to_csv(run / "table5_tier.csv", index=False)
        plot_confusion(m, cfg["tiers"], "Tier step on locked test set" +
                       (" (PRACTICE: not a result)" if cfg["mode"] == "practice" else ""), run / "confusion_tier.png")
    card = {"version_label": version_label, "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "feature_extractor": "PANNs CNN14 16 kHz (frozen)", "classifier_type": "Speech analyzer "
            "(timing measures + filler head + clarity head) with rule-based tier step",
            "criteria_version": crit["version"] if crit else None, "metric_names": METRIC_NAMES,
            "is_active": False, "mode": cfg["mode"], "label_file_version": cfg["gate"]["label_file_version"],
            "n_train": an.n_train, "test_scores": sc,
            "accuracy": sc.get("tier_accuracy"), "macro_f1": sc.get("tier_f1_macro"),
            "note": "Outputs speech metrics and a study-defined instructional prioritization tier. Not diagnostic; "
                    "does not measure words correct per minute or word accuracy."}
    an.save(run / "analyzer", card)
    lock.write_text(json.dumps({"opened_at": card["trained_at"], "run": str(run), "force_reason": force_reason}))
    append_run_log(cfg, run, "analyzer_final_test", {"version_label": version_label, "force_reason": force_reason,
                                                     "criteria_version": card["criteria_version"],
                                                     "filler_event_f1": sc.get("filler_event_f1"),
                                                     "clarity_mae": sc.get("clarity_mae"),
                                                     "tier_macro_f1": sc.get("tier_f1_macro")})
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in sc.items()}, indent=2))
    print(f"Analyzer final test -> {run}")
    return run


def analyze_file(cfg, run_dir, wav) -> dict:
    """One recording -> speech metrics + tier (what the Classification Service will return)."""
    import torch
    from . import audio as A
    from .analyzer import frame_features_for
    from .cnn14 import load_cnn14
    from .config import checkpoint_path
    from .embed import pick_device
    folder = Path(run_dir)
    if not (folder / "analyzer_card.json").exists():  # a run folder: the model is in its analyzer/ subfolder
        folder = folder / "analyzer"
    if not (folder / "analyzer_card.json").exists():
        raise FileNotFoundError(f"No saved analyzer in {run_dir} (analyzer_card.json not found)")
    an = SpeechAnalyzer.load(cfg, folder)
    ac, sr = cfg["audio"], cfg["audio"]["sample_rate"]
    y = A.load_audio(wav, sr)
    qc = A.quality_check(y, sr, ac["qc"], ac["trim_top_db"])
    if not qc.passed:
        return {"quality_status": "failed", "reasons": qc.reasons, "metrics": None, "tier": None}
    y = A.preprocess(y, ac)
    device = pick_device(cfg["cnn14"]["device"])
    cnn = load_cnn14(checkpoint_path(cfg, an.card["settings"]["cnn14_checkpoint"]), cfg["spectrogram"], sr, device)
    frames = frame_features_for(cnn, y, device, sr=sr)
    m, ev = an.analyze_arrays(y, frames)
    t = assign_tier(m, criteria(cfg))
    return {"quality_status": "passed", "metrics": m, "filler_events_s": ev, **t,
            "model_version": an.card["version_label"],
            "note": "Supplementary, non-diagnostic prioritization signal; the teacher decides."}
