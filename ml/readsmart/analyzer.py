"""The Read Smart 2.0 speech analyzer: one model that takes a recording and returns speech metrics.

Parts (all share the frozen CNN14 front end):
  - timing measures from the signal (pauses, speaking time, syllable-like peaks, rates): no labels
  - filler head: frame classifier on CNN14 frame features -> filler events ("uh", "um")
  - clarity head: ridge regression on pooled CNN14 frame features -> clarity score on the
    rater scale the team defines
The tier is NOT produced here; see tiers.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import medfilt
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cnn14 import load_cnn14
from .data import load_processed, usable_recordings, work
from .embed import pick_device
from .timing import timing_measures

FRAME_S = 0.08  # CNN14 block-3 frame step at 16 kHz / 10 ms hop


# ---------------------------------------------------------------------------
# Frame features (cached once per recording)
# ---------------------------------------------------------------------------
@torch.no_grad()
def frame_features_for(model, y: np.ndarray, device: str, chunk_s: float = 30.0, sr: int = 16000) -> np.ndarray:
    """Run CNN14 frame features over a whole recording in chunks (keeps memory flat for long files)."""
    n = int(chunk_s * sr)
    parts = []
    for s in range(0, len(y), n):
        seg = y[s:s + n]
        if len(seg) < sr // 2:  # pad very short tail chunks
            seg = np.pad(seg, (0, sr // 2 - len(seg)))
        f = model.frame_features(torch.from_numpy(seg)[None].to(device))[0].cpu().numpy()
        parts.append(f[: int(np.ceil(len(y[s:s + n]) / sr / FRAME_S))])
    return np.concatenate(parts).astype(np.float32)


def extract_frame_features(cfg) -> None:
    device = pick_device(cfg["cnn14"]["device"])
    sr = cfg["audio"]["sample_rate"]
    model = load_cnn14(cfg["cnn14"]["checkpoint"], cfg["spectrogram"], sr, device)
    rec = usable_recordings(cfg)
    for rid in rec.recording_id:
        np.save(work(cfg, "frames", f"{rid}.npy"), frame_features_for(model, load_processed(cfg, rid), device, sr=sr))
    print(f"Saved CNN14 frame features for {len(rec)} recordings ({FRAME_S * 1000:.0f} ms frames)")


def load_frames(cfg, rid) -> np.ndarray:
    return np.load(work(cfg, "frames", f"{rid}.npy"))


# ---------------------------------------------------------------------------
# Filler marks -> frame labels, and frame probabilities -> events
# ---------------------------------------------------------------------------
def frame_labels(n_frames: int, marks: list[tuple[float, float]]) -> np.ndarray:
    """1 where at least half of the 80 ms frame lies inside a marked filler."""
    t0 = np.arange(n_frames) * FRAME_S
    lab = np.zeros(n_frames, np.float32)
    for a, b in marks:
        ov = np.clip(np.minimum(t0 + FRAME_S, b) - np.maximum(t0, a), 0, None)
        lab[ov >= FRAME_S / 2] = 1
    return lab


def events_from_probs(p: np.ndarray, threshold: float, min_event_s: float, merge_gap_s: float):
    on = medfilt((p >= threshold).astype(float), 3) > 0.5
    ev, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            ev.append([start * FRAME_S, i * FRAME_S])
            start = None
    merged = []
    for e in ev:
        if merged and e[0] - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = e[1]
        else:
            merged.append(e)
    return [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= min_event_s]


class FillerNet(nn.Module):
    def __init__(self, d_in=768, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, hidden, 5, padding=2), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, 64, 5, padding=2), nn.ReLU(), nn.Conv1d(64, 1, 1))

    def forward(self, x):  # (B, T, D) -> (B, T)
        return self.net(x.transpose(1, 2))[:, 0, :]


class FillerHead:
    def __init__(self, fcfg: dict, seed: int, device="cpu"):
        self.c, self.seed, self.device = fcfg, seed, device

    def fit(self, seqs: list[np.ndarray], labels: list[np.ndarray]):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        allx = np.concatenate(seqs)
        self.mu, self.sd = allx.mean(0), allx.std(0) + 1e-6  # training data only
        ally = np.concatenate(labels)
        pos = max(ally.sum(), 1.0)
        self.net = FillerNet(seqs[0].shape[1], self.c["hidden"], self.c["dropout"]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.c["lr"], weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(ally) - pos) / pos, device=self.device))
        for _ in range(self.c["epochs"]):
            self.net.train()
            for i in rng.permutation(len(seqs)):
                x = torch.as_tensor((seqs[i] - self.mu) / self.sd, device=self.device)[None]
                y = torch.as_tensor(labels[i], device=self.device)[None]
                loss = lossf(self.net(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def frame_probs(self, x: np.ndarray) -> np.ndarray:
        self.net.eval()
        t = torch.as_tensor((x - self.mu) / self.sd, device=self.device)[None]
        return torch.sigmoid(self.net(t))[0].cpu().numpy()

    def events(self, x: np.ndarray):
        return events_from_probs(self.frame_probs(x), self.c["threshold"], self.c["min_event_s"], self.c["merge_gap_s"])


def pooled(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.mean(0), x.std(0)])


class ClarityHead:
    def __init__(self, ccfg: dict):
        self.c = ccfg
        self.est = make_pipeline(StandardScaler(), RidgeCV(alphas=ccfg["ridge_alphas"]))

    def fit(self, seqs, scores):
        self.est.fit(np.stack([pooled(s) for s in seqs]), np.asarray(scores, float))
        return self

    def predict(self, x: np.ndarray) -> float:
        v = float(self.est.predict(pooled(x)[None])[0])
        lo, hi = self.c["scale_min"], self.c["scale_max"]
        return float(np.clip(v, lo, hi)) if lo is not None and hi is not None else v


# ---------------------------------------------------------------------------
# The single analyzer
# ---------------------------------------------------------------------------
class SpeechAnalyzer:
    """Bundles the filler head, clarity head and timing settings. analyze() -> metrics dict."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.filler = FillerHead(cfg["filler"], cfg["seed"], pick_device(cfg["cnn14"]["device"]))
        self.clarity = ClarityHead(cfg["clarity"])

    def fit(self, rec: pd.DataFrame, marks: dict, train_learners):
        """rec: labels rows; marks: recording_id -> list of (start_s, end_s) in processed-audio time."""
        tr = rec[rec.learner_code.isin(set(train_learners))]
        fl = tr[tr.recording_id.isin(marks.keys())]
        if len(fl) == 0:
            raise ValueError("No filler-marked recordings among the training learners (filler marks are required).")
        seqs = [load_frames(self.cfg, r) for r in fl.recording_id]
        self.filler.fit(seqs, [frame_labels(len(s), marks[r]) for s, r in zip(seqs, fl.recording_id)])
        cl = tr[tr.clarity_score.astype(str).str.strip() != ""]
        if len(cl) < 3:
            raise ValueError("Fewer than 3 clarity-rated recordings among the training learners.")
        self.clarity.fit([load_frames(self.cfg, r) for r in cl.recording_id], cl.clarity_score.astype(float))
        self.n_train = {"filler_marked": int(len(fl)), "clarity_rated": int(len(cl))}
        return self

    def analyze_arrays(self, y: np.ndarray, frames: np.ndarray) -> dict:
        sr = self.cfg["audio"]["sample_rate"]
        ev = self.filler.events(frames)
        m = timing_measures(y, sr, self.cfg["timing"], exclude_spans=ev)
        dur = len(y) / sr
        m.update({
            "filler_count": len(ev),
            "fillers_per_min": round(len(ev) / (dur / 60), 2) if dur else 0.0,
            "filler_time_s": round(float(sum(b - a for a, b in ev)), 2),
            "clarity_score": round(self.clarity.predict(frames), 3),
        })
        return m, ev

    def analyze_recording(self, rid: str):
        return self.analyze_arrays(load_processed(self.cfg, rid), load_frames(self.cfg, rid))

    def save(self, folder: Path, card: dict):
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.filler.net.state_dict(), "mu": self.filler.mu, "sd": self.filler.sd},
                   folder / "filler_head.pt")
        joblib.dump(self.clarity.est, folder / "clarity_head.joblib")
        snap = {k: self.cfg[k] for k in ("audio", "spectrogram", "timing", "filler", "clarity", "tiers")}
        snap["cnn14_checkpoint"] = self.cfg["cnn14"]["checkpoint"]
        (folder / "analyzer_card.json").write_text(json.dumps({**card, "settings": snap}, indent=2, default=float))

    @classmethod
    def load(cls, cfg, folder: Path) -> "SpeechAnalyzer":
        card = json.loads((folder / "analyzer_card.json").read_text())
        s = card["settings"]
        cfg = dict(cfg)
        for k in ("timing", "filler", "clarity"):  # use the settings the analyzer was trained with
            cfg[k] = s[k]
        a = cls.__new__(cls)
        a.cfg, a.card = cfg, card
        saved = torch.load(folder / "filler_head.pt", weights_only=False)
        a.filler = FillerHead(s["filler"], cfg["seed"], pick_device(cfg["cnn14"]["device"]))
        a.filler.mu, a.filler.sd = saved["mu"], saved["sd"]
        a.filler.net = FillerNet(len(saved["mu"]), s["filler"]["hidden"], s["filler"]["dropout"])
        a.filler.net.load_state_dict(saved["state_dict"])
        a.clarity = ClarityHead(s["clarity"])
        a.clarity.est = joblib.load(folder / "clarity_head.joblib")
        return a


METRIC_NAMES = ["duration_s", "speaking_time_s", "speaking_ratio", "pause_count", "pauses_per_min",
                "total_pause_s", "mean_pause_s", "longest_pause_s", "syllable_count",
                "speech_rate_syl_per_s", "articulation_rate_syl_per_s",
                "filler_count", "fillers_per_min", "filler_time_s", "clarity_score"]
