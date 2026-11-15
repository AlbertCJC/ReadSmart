"""Waveform augmentation for TRAINING clips only: small gain change and low-level noise.
No pitch or speed change (Training Plan Section 7; Table 7 item 1 is still [DECIDE])."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .audio import load_audio


class WaveAugmenter:
    def __init__(self, aug_cfg: dict, sr: int, root: Path, seed: int):
        self.cfg = aug_cfg
        self.rng = np.random.default_rng(seed)
        self.noises = []
        if aug_cfg.get("noise_dir"):
            d = Path(aug_cfg["noise_dir"])
            d = d if d.is_absolute() else root / d
            self.noises = [load_audio(p, sr) for p in sorted(d.glob("*.wav"))]
            if not self.noises:
                raise FileNotFoundError(f"augment.noise_dir has no .wav files: {d}")

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        lo, hi = self.cfg["gain_db"]
        y = clip * 10 ** (self.rng.uniform(lo, hi) / 20)
        snr = self.rng.uniform(*self.cfg["noise_snr_db"])
        if self.noises:
            n = self.noises[self.rng.integers(len(self.noises))]
            if len(n) < len(y):
                n = np.tile(n, int(np.ceil(len(y) / len(n))))
            s = self.rng.integers(0, len(n) - len(y) + 1)
            n = n[s:s + len(y)]
        else:  # placeholder until classroom noise recordings exist
            n = self.rng.standard_normal(len(y)).astype(np.float32)
        p_sig = np.mean(y ** 2) + 1e-10
        p_noise = np.mean(n ** 2) + 1e-10
        y = y + n * np.sqrt(p_sig / (p_noise * 10 ** (snr / 10)))
        return np.clip(y, -1, 1).astype(np.float32)
