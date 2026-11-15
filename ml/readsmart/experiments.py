"""Cross-validation, model selection, and the single final test (Training Plan Sections 4, 6, 8)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from . import splits as S
from .augment import WaveAugmenter
from .data import clips_for, manual_exclusions, usable_recordings, work
from .embed import load_embeddings, pick_device
from .metrics import evaluate, plot_confusion, report_table
from .models import COMPLEXITY, HeadModel, ScratchCNNModel, SkModel, param_grid
from .runlog import new_run_dir, append_run_log


# ---------------------------------------------------------------------------
# Data views
# ---------------------------------------------------------------------------
class Data:
    """Holds everything the experiments need, loaded once."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.tiers = cfg["tiers"]
        self.tidx = {t: i for i, t in enumerate(self.tiers)}
        self.rec = usable_recordings(cfg)
        self.rec_learner = dict(zip(self.rec.recording_id, self.rec.learner_code))
        self.test, self.folds = S.load_splits(cfg)
        self.emb = {}
        self.hand = None
        self.clips = {}

    def embeddings(self, L):
        if L not in self.emb:
            self.emb[L] = load_embeddings(self.cfg, L)
        return self.emb[L]

    def handcrafted(self):
        if self.hand is None:
            self.hand = pd.read_csv(work(self.cfg, "handcrafted.csv"), dtype={"recording_id": str})
        return self.hand

    def wave_clips(self, L):
        """All kept clips of all usable recordings at length L (for M5)."""
        if L not in self.clips:
            ex = manual_exclusions(self.cfg)
            X, rid = [], []
            for r in self.rec.recording_id:
                kept, _ = clips_for(self.cfg, r, L, ex)
                X += [c for _, c in kept]
                rid += [r] * len(kept)
            self.clips[L] = (np.stack(X), np.array(rid))
        return self.clips[L]

    def y_of(self, recording_ids):
        tier = dict(zip(self.rec.recording_id, self.rec.reference_tier))
        return np.array([self.tidx[tier[r]] for r in recording_ids])

    def view(self, model, L, agg, learners, train: bool):
        """Rows for the given learners. Returns X, y, unit_ids (recording id of each row).
        Augmented copies (aug >= 1) are only returned for training (leakage checklist item 4)."""
        learners = set(learners)
        if model == "M1":
            h = self.handcrafted()
            h = h[h.recording_id.map(self.rec_learner).isin(learners)]
            return h.drop(columns="recording_id").to_numpy(np.float32), self.y_of(h.recording_id), h.recording_id.to_numpy()
        if model == "M5":
            X, rid = self.wave_clips(L)
            m = np.array([self.rec_learner[r] in learners for r in rid])
            return X[m], self.y_of(rid[m]), rid[m]
        X, meta = self.embeddings(L)
        m = meta.learner_code.isin(learners).to_numpy().copy()
        if not train:
            m &= (meta.aug == 0).to_numpy()
        X, meta = X[m], meta[m]
        if agg == "mean_embedding":
            g = meta.groupby(["recording_id", "aug"], sort=False).indices
            keys = list(g)
            X = np.stack([X[g[k]].mean(0) for k in keys])
            rid = np.array([k[0] for k in keys])
            return X, self.y_of(rid), rid
        return X, self.y_of(meta.recording_id), meta.recording_id.to_numpy()

    def aggregate(self, proba, unit_ids):
        """Average row probabilities per recording, then per learner if prediction_unit == learner."""
        df = pd.DataFrame(proba).assign(rid=unit_ids).groupby("rid", sort=True).mean()
        if self.cfg["split"]["prediction_unit"] == "learner":
            df = df.assign(lc=df.index.map(self.rec_learner)).groupby("lc").mean()
            lt = S.learner_table(self.rec, self.tiers).set_index("learner_code").reference_tier
            y = np.array([self.tidx[lt[u]] for u in df.index])
        else:
            y = self.y_of(df.index)
        return df.to_numpy(), y, df.index.to_numpy()


def make_model(cfg, name, params, augment_on: bool):
    seed, mcfg = cfg["seed"], cfg["models"][name]
    device = pick_device(cfg["cnn14"]["device"])
    if name in ("M0", "M1", "M2", "M3"):
        return SkModel(name, params, seed)
    if name == "M4":
        return HeadModel(params, mcfg, seed, device)
    if name == "M5":
        a = cfg["augment"]
        aug = WaveAugmenter(a, cfg["audio"]["sample_rate"], cfg.root, seed) if (augment_on and a["enabled"]) else None
        return ScratchCNNModel(params, mcfg, seed, device, cfg["spectrogram"], cfg["audio"]["sample_rate"],
                               aug, a["specaugment"] if (augment_on and a["enabled"]) else None)
    raise ValueError(name)


def inner_val_split(d: Data, learners, seed):
    """Hold out about a quarter of the training learners for early stopping (M4, M5).
    The evaluation fold is never used to stop training."""
    lt = S.learner_table(d.rec[d.rec.learner_code.isin(learners)], d.tiers)
    k = int(min(4, lt.reference_tier.value_counts().min()))
    if k < 2:
        return np.asarray(learners), np.array([])
    tr, va = next(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(
        lt, lt.reference_tier, groups=lt.learner_code))
    return lt.learner_code.iloc[tr].to_numpy(), lt.learner_code.iloc[va].to_numpy()


def fit_predict(d: Data, name, L, agg, params, train_l, eval_l, eval_is_test=False, epochs=None):
    cfg = d.cfg
    test_l = d.test.learner_code.to_numpy()
    S.check_no_leakage(train_l, eval_l, test_l, eval_is_test=eval_is_test)
    model = make_model(cfg, name, params, augment_on=True)
    if name in ("M4", "M5") and epochs is None:
        fit_l, stop_l = inner_val_split(d, train_l, cfg["seed"])
        S.check_no_leakage(fit_l, stop_l, test_l)
        Xtr, ytr, _ = d.view(name, L, agg, fit_l, train=True)
        Xva, yva, _ = d.view(name, L, agg, stop_l, train=False)
        model.fit(Xtr, ytr, Xva, yva)
    else:
        Xtr, ytr, _ = d.view(name, L, agg, train_l, train=True)
        model.fit(Xtr, ytr, epochs=epochs) if name in ("M4", "M5") else model.fit(Xtr, ytr)
    Xev, _, uid = d.view(name, L, agg, eval_l, train=False)
    proba, y, units = d.aggregate(model.predict_proba(Xev), uid)
    return model, proba, y, units


def configs(cfg):
    """Every (model, clip length, aggregation, params) combination to cross-validate."""
    out = []
    for name in ("M0", "M1", "M2", "M3", "M4", "M5"):
        mcfg = cfg["models"].get(name, {})
        if not mcfg.get("enabled"):
            continue
        if name == "M1":
            lengths, aggs = [None], ["recording"]
        elif name == "M5":  # optional per-model clip lengths, since M5 is slow on CPU
            lengths, aggs = mcfg.get("clip_lengths_s") or cfg["audio"]["clip_lengths_s"], ["mean_proba"]
        elif name == "M0":
            lengths, aggs = [cfg["audio"]["clip_lengths_s"][0]], ["mean_embedding"]
        else:
            lengths, aggs = cfg["audio"]["clip_lengths_s"], cfg["aggregation"]
        for L in lengths:
            for agg in aggs:
                for p in param_grid(name, mcfg):
                    out.append((name, L, agg, p))
    return out


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def run_cv(cfg, only: list[str] | None = None) -> Path:
    d = Data(cfg)
    run = new_run_dir(cfg, "cv")
    tiers = d.tiers
    rows = []
    nf = d.folds.fold.nunique()
    todo = [c for c in configs(cfg) if not only or c[0] in only]
    print(f"Cross-validating {len(todo)} configurations x {nf} folds")
    for name, L, agg, p in todo:
        for f in range(nf):
            tr, va = S.fold_learners(d.folds, f)
            model, proba, y, _ = fit_predict(d, name, L, agg, p, tr, va)
            m = evaluate(y, proba.argmax(1), tiers)
            rows.append({"model": name, "clip_s": L, "aggregation": agg, "params": json.dumps(p, sort_keys=True),
                         "fold": f, "best_epoch": getattr(model, "best_epoch", None),
                         **{k: v for k, v in m.items() if k != "confusion_matrix"},
                         "confusion_matrix": json.dumps(m["confusion_matrix"])})
        last = pd.DataFrame(rows[-nf:])
        print(f"  {name:3s} L={L!s:>4} {agg:15s} {p!s:40s} macroF1 {last.f1_macro.mean():.3f} ± {last.f1_macro.std():.3f}")
    folds = pd.DataFrame(rows)
    folds.to_csv(run / "cv_folds.csv", index=False)
    summary = summarize(folds, tiers)
    summary.to_csv(run / "cv_summary.csv", index=False)
    append_run_log(cfg, run, "cv", {"configs": len(todo), "folds": nf,
                                    "best_macro_f1": float(summary.f1_macro_mean.max())})
    print(f"\nCV results -> {run}")
    return run


def summarize(folds: pd.DataFrame, tiers) -> pd.DataFrame:
    keys = ["model", "clip_s", "aggregation", "params"]
    folds = folds.assign(clip_s=[("None" if pd.isna(v) else str(v)) for v in folds.clip_s])  # M1 has no clip length
    metrics = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"] + \
              [f"recall_{t}" for t in tiers] + [f"f1_{t}" for t in tiers]
    g = folds.groupby(keys, sort=False)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["median_best_epoch"] = g.best_epoch.median()
    return out.reset_index().sort_values("f1_macro_mean", ascending=False)


# ---------------------------------------------------------------------------
# Model selection (Section 8.2): written down before the test set is opened
# ---------------------------------------------------------------------------
def latest_summaries(cfg, cv_runs: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Combine CV summaries; for each model keep the rows from the most recent run that has it.
    This lets slow models (M5) be cross-validated in a separate run on the same fold file."""
    parts, used = [], {}
    for run in sorted(cv_runs, key=lambda p: p.name, reverse=True):
        if not (run / "cv_summary.csv").exists():  # interrupted run
            continue
        s = pd.read_csv(run / "cv_summary.csv", dtype={"clip_s": str})
        s = s[~s.model.isin(used)]
        for m in s.model.unique():
            used[m] = str(run)
        parts.append(s)
    return pd.concat(parts, ignore_index=True), used


def select(cfg, cv_runs: list[Path], eligible=("M2", "M3", "M4")) -> dict:
    s, used = latest_summaries(cfg, cv_runs)
    tiers = cfg["tiers"]
    high = f"recall_{tiers[-1]}_mean"
    floor = cfg["gate"]["min_high_recall"]
    best_per_model = s.sort_values("f1_macro_mean", ascending=False).groupby("model").head(1)
    cand = best_per_model[best_per_model.model.isin(eligible)].copy()
    notes = []
    if floor is None:
        notes.append("gate.min_high_recall is not set ([TO BE FILLED] G8); High-recall floor NOT applied.")
    else:
        passed = cand[cand[high] >= floor]
        if passed.empty:
            notes.append(f"No eligible model meets High recall >= {floor}; choosing by macro F1 and flagging it.")
        else:
            cand = passed
    if cand.empty:
        raise ValueError(f"No eligible model in CV results (eligible: {eligible})")
    top = cand.sort_values("f1_macro_mean", ascending=False).iloc[0]
    close = cand[cand.f1_macro_mean >= top.f1_macro_mean - top.f1_macro_std]
    chosen = close.assign(c=close.model.map(COMPLEXITY)).sort_values(["c", "f1_macro_mean"], ascending=[True, False]).iloc[0]
    if chosen.model != top.model:
        notes.append(f"{chosen.model} chosen over {top.model}: within one SD of the best macro F1, and simpler.")
    baselines = {}
    for b in ("M0", "M1", "M5"):
        r = best_per_model[best_per_model.model == b]
        if len(r):
            baselines[b] = _cfg_of(r.iloc[0])
    decision = {
        "written_at": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": cfg["mode"], "cv_runs_by_model": used,
        "rule": "best mean CV macro F1 among eligible models, subject to the High-recall floor; "
                "a simpler model within one SD of the best is preferred (Training Plan 8.2)",
        "chosen": _cfg_of(chosen), "baselines": baselines, "notes": notes,
        "candidates": cand[["model", "clip_s", "aggregation", "params", "f1_macro_mean", "f1_macro_std", high]].to_dict("records"),
    }
    path = work(cfg, "decision.json")
    path.write_text(json.dumps(decision, indent=2))
    print(json.dumps({k: decision[k] for k in ("chosen", "baselines", "notes")}, indent=2))
    print(f"Decision written to {path}. Record it in the sprint notes before running the final test.")
    return decision


def _cfg_of(row) -> dict:
    L = None if pd.isna(row.clip_s) or row.clip_s in ("None", "nan") else float(row.clip_s)
    ep = None if pd.isna(row.median_best_epoch) else int(round(row.median_best_epoch))
    return {"model": row.model, "clip_s": L, "aggregation": row.aggregation,
            "params": json.loads(row.params), "epochs": ep, "cv_macro_f1": float(row.f1_macro_mean)}


# ---------------------------------------------------------------------------
# Final test (Section 8.3): retrain on all development learners, test once
# ---------------------------------------------------------------------------
def final_test(cfg, version_label: str, force_reason: str | None = None) -> Path:
    lock = work(cfg, "splits", "TEST_SET_OPENED.json")
    if lock.exists() and not force_reason:
        raise RuntimeError(f"The locked test set was already used ({lock}). It is used once. "
                           "Re-running needs --force-reason, which is written to the run log.")
    decision = json.loads(work(cfg, "decision.json").read_text())
    d = Data(cfg)
    run = new_run_dir(cfg, "final_test")
    dev, test = S.all_dev_learners(d.folds), d.test.learner_code.to_numpy()
    results = {}
    entries = [("chosen", decision["chosen"])] + list(decision["baselines"].items())
    for role, c in entries:
        L = c["clip_s"] if c["clip_s"] is None else (int(c["clip_s"]) if float(c["clip_s"]).is_integer() else c["clip_s"])
        model, proba, y, units = fit_predict(d, c["model"], L, c["aggregation"], c["params"], dev, test,
                                             eval_is_test=True, epochs=c.get("epochs"))
        m = evaluate(y, proba.argmax(1), d.tiers)
        results[role] = {"config": c, "metrics": m}
        tag = f"{role}_{c['model']}"
        report_table(m, d.tiers).to_csv(run / f"table5_{tag}.csv", index=False)
        plot_confusion(m, d.tiers, f"{c['model']} on locked test set" +
                       (" (PRACTICE: not a result)" if cfg["mode"] == "practice" else ""), run / f"confusion_{tag}.png")
        pd.DataFrame(proba, columns=d.tiers).assign(unit=units, reference=[d.tiers[i] for i in y]).to_csv(
            run / f"predictions_{tag}.csv", index=False)
        print(f"{role:9s} {c['model']}: accuracy {m['accuracy']:.3f}  macro F1 {m['f1_macro']:.3f}  "
              f"High recall {m['recall_' + d.tiers[-1]]:.3f}")
        if role == "chosen":
            save_model(cfg, run, model, c, m, version_label)
    (run / "test_results.json").write_text(json.dumps(results, indent=2, default=float))
    lock.write_text(json.dumps({"opened_at": dt.datetime.now().isoformat(timespec="seconds"),
                                "run": str(run), "force_reason": force_reason}, indent=2))
    target = cfg["gate"]["target_macro_f1"]
    met = None if target is None else results["chosen"]["metrics"]["f1_macro"] >= target
    append_run_log(cfg, run, "final_test", {"version_label": version_label,
                                            "macro_f1": results["chosen"]["metrics"]["f1_macro"],
                                            "target_met": met, "force_reason": force_reason})
    print(f"Final test results -> {run}  (Validation Board S1 target met: {met})")
    return run


def save_model(cfg, run: Path, model, c: dict, m: dict, version_label: str) -> None:
    """Model file + model card with MODEL_VERSION fields (Chapter III Table 3.19) for Module 9."""
    if c["model"] in ("M4", "M5"):
        torch.save({"state_dict": model.net.state_dict(), "mu": getattr(model, "mu", None),
                    "sd": getattr(model, "sd", None)}, run / "model.pt")
        fname = "model.pt"
    else:
        joblib.dump(model.est, run / "model.joblib")
        fname = "model.joblib"
    names = {"M1": "Logistic regression (hand-crafted measures)", "M2": "Logistic regression",
             "M3": "Support vector machine", "M4": "Small neural head", "M5": "Small CNN from scratch"}
    card = {
        "version_label": version_label,
        "feature_extractor": "PANNs CNN14 16 kHz (frozen)" if c["model"] in ("M2", "M3", "M4") else "none",
        "cnn14_checkpoint": cfg["cnn14"]["checkpoint"],
        "classifier_type": names.get(c["model"], c["model"]),
        "accuracy": round(m["accuracy"], 4), "macro_f1": round(m["f1_macro"], 4),
        "is_active": False,  # set active through Module 9 only
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_mels": cfg["spectrogram"]["n_mels"], "clip_s": c["clip_s"], "aggregation": c["aggregation"],
        "params": c["params"], "tiers": cfg["tiers"], "confusion_matrix": m["confusion_matrix"],
        "per_tier": {t: {"precision": m[f"precision_{t}"], "recall": m[f"recall_{t}"], "f1": m[f"f1_{t}"]}
                     for t in cfg["tiers"]},
        "model_file": fname, "mode": cfg["mode"],
        "label_file_version": cfg["gate"]["label_file_version"],
        "note": "Outputs a study-defined instructional prioritization tier. Not diagnostic.",
    }
    (run / "model_card.json").write_text(json.dumps(card, indent=2, default=float))
