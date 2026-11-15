"""Pass every kept clip through frozen CNN14 once and cache the 2,048-d embeddings (Section 5).

Cache per clip length: work/embeddings/emb_{L}s.npz with
  X    (N, 2048) float32
  meta columns: recording_id, learner_code, clip_index, aug, reference_tier
aug = 0 is the original clip; aug >= 1 are augmented copies, which are only ever
used when that learner is in the training folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .augment import WaveAugmenter
from .cnn14 import load_cnn14, spec_mask
from .data import clips_for, manual_exclusions, usable_recordings, work


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


@torch.no_grad()
def _embed_batch(model, clips, device, mask_fn=None):
    x = torch.from_numpy(np.stack(clips)).to(device)
    return model(x, mask_fn=mask_fn).cpu().numpy()


def extract_embeddings(cfg) -> None:
    device = pick_device(cfg["cnn14"]["device"])
    sr = cfg["audio"]["sample_rate"]
    model = load_cnn14(cfg["cnn14"]["checkpoint"], cfg["spectrogram"], sr, device)
    rec = usable_recordings(cfg)
    excluded = manual_exclusions(cfg)
    aug_cfg = cfg["augment"]
    copies = aug_cfg["embedding_copies"] if aug_cfg["enabled"] else 0
    augmenter = WaveAugmenter(aug_cfg, sr, cfg.root, cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    sa = aug_cfg["specaugment"]
    mask_fn = lambda x: spec_mask(x, sa["time_masks"], sa["time_width"], sa["freq_masks"], sa["freq_width"], gen)
    bs = cfg["cnn14"]["batch_size"]

    for L in cfg["audio"]["clip_lengths_s"]:
        X, meta = [], []
        for r in rec.itertuples():
            kept, _ = clips_for(cfg, r.recording_id, L, excluded)
            idx = [i for i, _ in kept]
            wav = [c for _, c in kept]
            for a in range(copies + 1):
                batch_in = wav if a == 0 else [augmenter(c) for c in wav]
                for s in range(0, len(batch_in), bs):
                    X.append(_embed_batch(model, batch_in[s:s + bs], device, None if a == 0 else mask_fn))
                meta += [(r.recording_id, r.learner_code, i, a, r.reference_tier) for i in idx]
        X = np.concatenate(X).astype(np.float32)
        meta = pd.DataFrame(meta, columns=["recording_id", "learner_code", "clip_index", "aug", "reference_tier"])
        out = work(cfg, "embeddings", f"emb_{L}s.npz")
        np.savez_compressed(out, X=X, **{c: np.asarray(meta[c].astype(str).tolist(), dtype=str) for c in meta.columns})
        print(f"Clip length {L}s: {int((meta.aug == 0).sum())} clips, {len(meta)} rows with augmentation -> {out}")


def load_embeddings(cfg, L) -> tuple[np.ndarray, pd.DataFrame]:
    z = np.load(work(cfg, "embeddings", f"emb_{L}s.npz"), allow_pickle=False)
    meta = pd.DataFrame({c: z[c] for c in ["recording_id", "learner_code", "clip_index", "aug", "reference_tier"]})
    meta["aug"] = meta.aug.astype(int)
    meta["clip_index"] = meta.clip_index.astype(int)
    return z["X"], meta
