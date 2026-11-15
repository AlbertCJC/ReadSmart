"""Command line: python -m readsmart <command> [--config config.yaml]

Order of use (Training Plan Table 6):
  gate        list open readiness-gate items
  qc          quality check + preprocessing of every recording
  clips       segment into clips for each candidate clip length, log dropped clips
  handcrafted M1 simple acoustic measures (only if models.M1.enabled)
  embed       cache frozen CNN14 embeddings (+ augmented copies for training)
  split       lock the learner-level test set and save the fold file
  inventory   print Training Plan Table 2
  kappa       rater agreement on the double-labeled subset
  cv          cross-validate every model/config on the folds
  select      choose the final model by the rule in Section 8.2 (written to work/decision.json)
  final       retrain on all development learners and test ONCE on the locked test set
  predict     classify one WAV with a saved model
  figure      save a sample log-mel spectrogram image
  practice-all  run qc..final in practice mode (pipeline check only)

Speech analyzer (one model -> speech metrics; separate tier step from criteria/tier_criteria.yaml):
  import-audacity  convert Audacity label files (--labels-dir) into data/filler_marks.csv
  frames           cache CNN14 frame features for every usable recording
  timing-check     compare automatic pauses / syllable peaks with hand-timed values (dev learners)
  criteria-check   validate the tier criteria file
  analyzer-cv      cross-validate the filler and clarity heads (+ tier step if criteria are set)
  analyzer-final   train on all development learners, test ONCE on the locked test set, save the model
  analyze          one WAV -> speech metrics + tier (--run <analyzer final run>, --wav)
  practice-analyzer  run qc, split, frames, analyzer-cv, analyzer-final in practice mode
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import check_gate, gate_problems, load_config, set_seeds


def main(argv=None):
    ap = argparse.ArgumentParser(prog="readsmart", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="redraw the test set / folds (split only)")
    ap.add_argument("--force-reason", help="reason for re-opening the test set (final only; logged)")
    ap.add_argument("--version-label", default="v0-practice", help="MODEL_VERSION.version_label (final)")
    ap.add_argument("--models", help="comma-separated subset for cv, e.g. M0,M2")
    ap.add_argument("--cv-run", help="comma-separated CV run folders for select (default: all CV runs, latest per model)")
    ap.add_argument("--run", "--model", dest="run", help="saved model: a final-test run folder, or an analyzer folder (analyze / predict)")
    ap.add_argument("--wav", help="WAV file for predict / figure")
    ap.add_argument("--double-labels", default="data/double_labels.csv")
    ap.add_argument("--labels-dir", help="folder of Audacity label .txt files (import-audacity)")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    set_seeds(cfg["seed"])
    cmd = a.command

    if cmd == "gate":
        probs = gate_problems(cfg)
        print("Readiness gate: all closed." if not probs else "Open items:\n  - " + "\n  - ".join(probs))
        return
    check_gate(cfg)

    from . import data, experiments, splits
    if cmd in ("qc", "practice-all"):
        data.run_qc_and_preprocess(cfg)
    if cmd in ("clips", "practice-all"):
        log = data.clip_index(cfg)
        print(log.groupby("clip_s")[["clips_total", "clips_kept", "clips_dropped"]].sum().to_string())
    if cmd in ("handcrafted", "practice-all") and cfg["models"]["M1"]["enabled"]:
        data.handcrafted_table(cfg)
        print("Saved M1 hand-crafted measures.")
    if cmd in ("embed", "practice-all"):
        from .embed import extract_embeddings
        extract_embeddings(cfg)
    if cmd in ("split", "practice-all"):
        splits.lock_test_set(cfg, force=a.force or cmd == "practice-all")
        splits.make_folds(cfg, force=True)
    if cmd in ("inventory", "practice-all"):
        print("\nTable 2. Dataset inventory\n" + data.inventory(cfg).to_string())
    if cmd == "kappa":
        p = Path(a.double_labels)
        p = p if p.is_absolute() else cfg.root / p
        dl = pd.read_csv(p, dtype=str)
        from .metrics import rater_agreement
        print(json.dumps(rater_agreement(dl.rater_a_tier, dl.rater_b_tier, cfg["tiers"]), indent=2))
    cv_run = None
    if cmd in ("cv", "practice-all"):
        cv_run = experiments.run_cv(cfg, a.models.split(",") if a.models else None)
    if cmd in ("select", "practice-all"):
        if a.cv_run:
            runs = [Path(r) for r in a.cv_run.split(",")]
        elif cv_run is not None:
            runs = [cv_run]
        else:  # every CV run of this mode; the latest run per model is used
            runs = sorted(cfg.path("runs_dir").glob(f"*_{cfg['mode']}_cv"))
        if not runs:
            sys.exit("No CV run found. Run `cv` first.")
        experiments.select(cfg, runs)
    if cmd in ("final", "practice-all"):
        reason = a.force_reason or ("practice-all rerun" if cmd == "practice-all" else None)
        experiments.final_test(cfg, a.version_label, reason)
    if cmd == "predict":
        from .predict import predict_file
        print(json.dumps(predict_file(cfg, a.run, a.wav), indent=2))
    if cmd == "figure":
        save_logmel_figure(cfg, a.wav)
    if cmd == "practice-analyzer":
        if cfg["mode"] != "practice":
            sys.exit("practice-analyzer is for practice mode only")
        data.run_qc_and_preprocess(cfg)
        splits.lock_test_set(cfg, force=True)
        splits.make_folds(cfg, force=True)
    from . import analyzer_eval as AE
    if cmd == "import-audacity":
        data.import_audacity_labels(cfg, a.labels_dir)
    if cmd in ("frames", "practice-analyzer"):
        from .analyzer import extract_frame_features
        extract_frame_features(cfg)
    if cmd in ("timing-check", "practice-analyzer"):
        AE.timing_check(cfg)
    if cmd == "criteria-check":
        c = AE.criteria(cfg)
        print("Criteria not set: every result will say 'criteria not set'." if c is None
              else f"Criteria OK: version {c['version']}, {len(c['rules'])} rules, default {c['default_tier']}")
    if cmd in ("analyzer-cv", "practice-analyzer"):
        AE.analyzer_cv(cfg)
    if cmd in ("analyzer-final", "practice-analyzer"):
        reason = a.force_reason or ("practice-analyzer rerun" if cmd == "practice-analyzer" else None)
        AE.analyzer_final(cfg, a.version_label, reason)
    if cmd == "analyze":
        print(json.dumps(AE.analyze_file(cfg, a.run, a.wav), indent=2, default=float))
    known = {"import-audacity", "frames", "timing-check", "criteria-check", "analyzer-cv", "analyzer-final",
             "analyze", "practice-analyzer", "qc", "clips", "handcrafted", "embed", "split", "inventory", "kappa", "cv", "select",
             "final", "predict", "figure", "practice-all"}
    if cmd not in known:
        sys.exit(f"Unknown command {cmd}. See: python -m readsmart --help")


def save_logmel_figure(cfg, wav):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from . import audio as A
    from .cnn14 import LogMel
    sc, sr = cfg["spectrogram"], cfg["audio"]["sample_rate"]
    y = A.preprocess(A.load_audio(wav, sr), cfg["audio"])
    lm = LogMel(sr, sc["n_fft"], sc["hop_length"], sc["n_mels"], sc["fmin"], sc["fmax"])
    with torch.no_grad():
        S = lm(torch.from_numpy(y)[None])[0, 0].numpy().T
    fig, ax = plt.subplots(figsize=(7, 2.6), dpi=200)
    im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, len(y) / sr, 0, sc["n_mels"]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel band")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    out = Path(wav).with_suffix(".logmel.png")
    fig.savefig(out)
    print(f"Saved {out} (use anonymized learner codes only in figure captions)")


if __name__ == "__main__":
    main()
