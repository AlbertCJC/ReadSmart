"""Audio loading, quality check, preprocessing, segmentation, and simple acoustic measures.

Preprocessing follows Training Plan Table 3: mono, 16 kHz, trim leading/trailing
silence, normalize loudness, cut fixed-length overlapping clips, drop clips that
are mostly silence.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import librosa
import numpy as np
import soundfile as sf

EPS = 1e-10


def db(x: float) -> float:
    return float(20 * np.log10(max(x, EPS)))


def load_audio(path, sr: int) -> np.ndarray:
    """Read any soundfile-supported file, mix to mono, resample to sr, float32 in [-1, 1]."""
    y, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
    y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y.astype(np.float32)


@dataclass
class QCResult:
    passed: bool
    duration_s: float
    rms_dbfs: float
    clipped_fraction: float
    reasons: str

    def as_dict(self):
        return asdict(self)


def quality_check(y_raw: np.ndarray, sr: int, qc_cfg: dict, trim_top_db: float) -> QCResult:
    """Minimum duration, minimum loudness, no clipping (Chapter III Program Workflow step 6)."""
    clipped = float(np.mean(np.abs(y_raw) >= 0.999)) if len(y_raw) else 1.0
    y = trim(y_raw, trim_top_db)
    dur = len(y) / sr
    rms = db(np.sqrt(np.mean(y ** 2))) if len(y) else -120.0
    reasons = []
    if dur < qc_cfg["min_duration_s"]:
        reasons.append(f"too short ({dur:.1f}s < {qc_cfg['min_duration_s']}s)")
    if rms < qc_cfg["min_rms_dbfs"]:
        reasons.append(f"too quiet ({rms:.1f} dBFS < {qc_cfg['min_rms_dbfs']})")
    if clipped > qc_cfg["max_clipped_fraction"]:
        reasons.append(f"clipping ({clipped:.4f} of samples at full scale)")
    return QCResult(not reasons, round(dur, 2), round(rms, 2), round(clipped, 5), "; ".join(reasons))


def trim(y: np.ndarray, top_db: float) -> np.ndarray:
    if len(y) == 0:
        return y
    yt, _ = librosa.effects.trim(y, top_db=top_db, frame_length=512, hop_length=160)
    return yt


def trim_start_s(y: np.ndarray, top_db: float, sr: int) -> float:
    """Seconds removed from the start by trim(); used to shift hand-made time marks."""
    if len(y) == 0:
        return 0.0
    _, idx = librosa.effects.trim(y, top_db=top_db, frame_length=512, hop_length=160)
    return float(idx[0]) / sr


def normalize_loudness(y: np.ndarray, target_dbfs: float) -> np.ndarray:
    rms = np.sqrt(np.mean(y ** 2)) + EPS
    y = y * (10 ** (target_dbfs / 20) / rms)
    peak = np.max(np.abs(y)) + EPS
    if peak > 0.99:  # avoid creating clipping
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def preprocess(y_raw: np.ndarray, audio_cfg: dict) -> np.ndarray:
    return normalize_loudness(trim(y_raw, audio_cfg["trim_top_db"]), audio_cfg["target_rms_dbfs"])


def active_mask(y: np.ndarray, top_db: float, hop: int = 160) -> np.ndarray:
    """Per-frame True where the frame is within top_db of the loudest frame of the recording."""
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop, center=True)[0]
    ref = rms.max() + EPS
    return 20 * np.log10(rms / ref + EPS) > -top_db


def segment(y: np.ndarray, sr: int, clip_s: float, overlap: float, top_db: float,
            min_active: float, hop: int = 160):
    """Cut fixed-length clips. Returns list of (clip_index, start_s, clip, active_fraction, kept)."""
    n = int(round(clip_s * sr))
    step = max(1, int(round(n * (1 - overlap))))
    mask = active_mask(y, top_db, hop)
    if len(y) <= n:
        starts = [0]
    else:
        starts = list(range(0, len(y) - n + 1, step))
        if starts[-1] + n < len(y):  # keep the tail as a final clip
            starts.append(len(y) - n)
    out = []
    for i, s in enumerate(starts):
        clip = y[s:s + n]
        if len(clip) < n:
            clip = np.pad(clip, (0, n - len(clip)))
        f0, f1 = s // hop, (s + n) // hop + 1
        frac = float(mask[f0:f1].mean()) if f1 > f0 else 0.0
        out.append((i, s / sr, clip.astype(np.float32), frac, frac >= min_active))
    return out


# ---------------------------------------------------------------------------
# M1: simple acoustic measures (optional baseline; Training Plan Table 4, Table 7 item 5)
# These are recording-level measures of the kind used in prosodic studies of
# children's read speech (Dimzon [VERIFY year]). They are NOT words correct per
# minute or word accuracy; onset rate is only a rough speech-rate proxy.
# ---------------------------------------------------------------------------
HANDCRAFTED_NAMES = [
    "duration_s", "active_ratio", "pauses_per_min", "mean_pause_s", "max_pause_s",
    "onset_rate", "pitch_mean_hz", "pitch_sd_hz", "pitch_range_hz",
    "rms_sd_db", "centroid_mean_hz", "centroid_sd_hz",
]


def handcrafted_features(y: np.ndarray, sr: int, top_db: float, min_pause_s: float = 0.25) -> np.ndarray:
    hop = 160
    mask = active_mask(y, top_db, hop)
    frame_s = hop / sr
    dur = len(y) / sr
    # pauses = runs of inactive frames longer than min_pause_s
    pauses, run = [], 0
    for m in mask:
        if not m:
            run += 1
        else:
            if run * frame_s >= min_pause_s:
                pauses.append(run * frame_s)
            run = 0
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="time")
    f0 = librosa.yin(y, fmin=75, fmax=500, sr=sr, frame_length=1024, hop_length=hop)
    n = min(len(f0), len(mask))
    f0 = f0[:n][mask[:n]]
    if len(f0) == 0:
        f0 = np.array([0.0])
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop)[0]
    rms_db = 20 * np.log10(rms + EPS)[mask[: len(rms)]] if mask.any() else np.array([0.0])
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=hop)[0]
    feats = [
        dur, float(mask.mean()), len(pauses) / max(dur / 60, EPS),
        float(np.mean(pauses)) if pauses else 0.0, float(np.max(pauses)) if pauses else 0.0,
        len(onsets) / max(dur, EPS),
        float(np.median(f0)), float(np.std(f0)), float(np.percentile(f0, 95) - np.percentile(f0, 5)),
        float(np.std(rms_db)), float(np.mean(cent)), float(np.std(cent)),
    ]
    return np.asarray(feats, dtype=np.float32)
