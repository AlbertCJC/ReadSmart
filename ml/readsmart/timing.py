"""Signal-based timing measures: pauses, speaking time, syllable-like peaks, speech and articulation rate.

These need no training labels. They are checked against hand-timed values on a small set of
recordings (`timing-check`). Syllable peaks follow the idea of counting intensity peaks in voiced
speech that stand out from the surrounding dips (de Jong & Wempe, 2009 [VERIFY citation before
using it in the thesis]). They are a rough rate measure, not word counts, and not words correct
per minute.
"""
from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import find_peaks, medfilt

HOP = 160  # 10 ms at 16 kHz


def frame_db(y: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=HOP, center=True)[0]
    return 20 * np.log10(rms + 1e-10)


def speech_mask(y: np.ndarray, silence_db: float) -> tuple[np.ndarray, float]:
    """True where a 10 ms frame is speech-like: within silence_db of the loud level of the recording
    (95th percentile of frame loudness, so a single loud click does not set the reference)."""
    d = frame_db(y)
    ref = np.percentile(d, 95)
    m = d > ref - silence_db
    m = medfilt(m.astype(float), 5) > 0.5  # remove 10-20 ms flickers
    return m, ref


def runs(mask: np.ndarray, value: bool):
    """(start_frame, end_frame) of consecutive runs equal to value."""
    out, start = [], None
    for i, v in enumerate(np.append(mask, not value)):
        if v == value and start is None:
            start = i
        elif v != value and start is not None:
            out.append((start, i))
            start = None
    return out


def pauses(y: np.ndarray, sr: int, silence_db: float, min_pause_s: float):
    """Silent stretches of at least min_pause_s between speech. Leading/trailing silence is not counted
    (it was trimmed in preprocessing, and any left is excluded here)."""
    m, _ = speech_mask(y, silence_db)
    fs = HOP / sr
    speech = np.flatnonzero(m)
    if len(speech) == 0:
        return [], m
    inner = m[speech[0]:speech[-1] + 1]
    off = speech[0]
    ps = [((a + off) * fs, (b + off) * fs) for a, b in runs(inner, False) if (b - a) * fs >= min_pause_s]
    return ps, m


def syllable_peaks(y: np.ndarray, sr: int, silence_db: float, min_dip_db: float, voiced_only: bool = True):
    """Times (s) of syllable-like intensity peaks in voiced speech."""
    d = frame_db(y)
    m, ref = speech_mask(y, silence_db)
    smooth = np.convolve(d, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(smooth, prominence=min_dip_db, distance=int(0.08 * sr / HOP))
    peaks = peaks[m[np.minimum(peaks, len(m) - 1)]]
    if voiced_only and len(peaks):
        _, voiced, _ = librosa.pyin(y, fmin=75, fmax=500, sr=sr, frame_length=1024, hop_length=HOP)
        voiced = np.nan_to_num(voiced).astype(bool)
        peaks = peaks[voiced[np.minimum(peaks, len(voiced) - 1)]]
    return peaks * HOP / sr


def timing_measures(y: np.ndarray, sr: int, cfg: dict, exclude_spans=()) -> dict:
    """cfg = config['timing']. exclude_spans: detected filler events (start_s, end_s); syllable peaks
    inside them are not counted as syllables, and filler time counts as speaking time but not as
    articulation time."""
    ps, m = pauses(y, sr, cfg["silence_db"], cfg["min_pause_s"])
    dur = len(y) / sr
    total_pause = float(sum(b - a for a, b in ps))
    speaking = max(dur - total_pause, 1e-6)
    peaks = syllable_peaks(y, sr, cfg["silence_db"], cfg["min_dip_db"], cfg.get("voiced_only", True))
    if exclude_spans:
        keep = np.ones(len(peaks), bool)
        for a, b in exclude_spans:
            keep &= ~((peaks >= a) & (peaks <= b))
        peaks = peaks[keep]
    filler_time = float(sum(b - a for a, b in exclude_spans))
    artic_time = max(speaking - filler_time, 1e-6)
    return {
        "duration_s": round(dur, 2),
        "speaking_time_s": round(speaking, 2),
        "speaking_ratio": round(speaking / dur, 4) if dur else 0.0,
        "pause_count": len(ps),
        "pauses_per_min": round(len(ps) / (dur / 60), 2) if dur else 0.0,
        "total_pause_s": round(total_pause, 2),
        "mean_pause_s": round(total_pause / len(ps), 3) if ps else 0.0,
        "longest_pause_s": round(max((b - a for a, b in ps), default=0.0), 3),
        "syllable_count": int(len(peaks)),
        "speech_rate_syl_per_s": round(len(peaks) / dur, 3) if dur else 0.0,
        "articulation_rate_syl_per_s": round(len(peaks) / artic_time, 3),
    }
