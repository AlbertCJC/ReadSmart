"""Classify one recording with a saved model (what the Classification Service will do)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from . import audio as A
from .cnn14 import load_cnn14
from .config import checkpoint_path
from .embed import pick_device
from .models import HeadModel


def predict_file(cfg, run_dir: str | Path, wav_path: str | Path) -> dict:
    run_dir = Path(run_dir)
    card = json.loads((run_dir / "model_card.json").read_text())
    if card["feature_extractor"] == "none":
        raise NotImplementedError("predict supports the CNN14-based models (M2-M4) only.")
    ac, sr = cfg["audio"], cfg["audio"]["sample_rate"]
    y = A.load_audio(wav_path, sr)
    qc = A.quality_check(y, sr, ac["qc"], ac["trim_top_db"])
    if not qc.passed:
        return {"quality_status": "failed", "reasons": qc.reasons, "tier": None}
    y = A.preprocess(y, ac)
    L = card["clip_s"]
    segs = A.segment(y, sr, L, ac["clip_overlap"], ac["trim_top_db"], ac["clip_min_active_fraction"])
    clips = [c for _, _, c, _, k in segs if k] or [max(segs, key=lambda s: s[3])[2]]
    device = pick_device(cfg["cnn14"]["device"])
    cnn = load_cnn14(checkpoint_path(cfg, card["cnn14_checkpoint"]), cfg["spectrogram"], sr, device)
    with torch.no_grad():
        emb = cnn(torch.from_numpy(np.stack(clips)).to(device)).cpu().numpy()
    X = emb.mean(0, keepdims=True) if card["aggregation"] == "mean_embedding" else emb
    if card["model_file"] == "model.joblib":
        est = joblib.load(run_dir / "model.joblib")
        p = np.zeros((len(X), len(card["tiers"])))
        p[:, est.classes_] = est.predict_proba(X)
    else:
        saved = torch.load(run_dir / "model.pt", weights_only=False)
        mcfg = cfg["models"]["M4"]
        head = HeadModel(card["params"], mcfg, cfg["seed"], device)
        head.mu, head.sd = saved["mu"], saved["sd"]
        head.net = torch.nn.Sequential(
            torch.nn.Dropout(0), torch.nn.Linear(X.shape[1], mcfg["hidden"]), torch.nn.ReLU(),
            torch.nn.Dropout(0), torch.nn.Linear(mcfg["hidden"], len(card["tiers"])))
        head.net.load_state_dict(saved["state_dict"])
        p = head.predict_proba(X)
    p = p.mean(0)
    return {"quality_status": "passed", "tier": card["tiers"][int(p.argmax())],
            "probabilities": {t: round(float(v), 4) for t, v in zip(card["tiers"], p)},
            "model_version": card["version_label"],
            "note": "Supplementary, non-diagnostic prioritization signal; the teacher decides."}
