"""Load config.yaml and enforce the readiness gate in study mode."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml


class GateError(RuntimeError):
    pass


class Config(dict):
    """dict with attribute access and a few path helpers."""

    def __getattr__(self, key):
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    @property
    def root(self) -> Path:
        return Path(self["_root"])

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else self.root / p


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with open(path) as f:
        cfg = Config(yaml.safe_load(f))
    cfg["_root"] = str(path.parent)
    cfg["_path"] = str(path)
    if cfg["mode"] not in ("practice", "study"):
        raise ValueError("mode must be 'practice' or 'study'")
    if cfg["augment"].get("pitch_shift") or cfg["augment"].get("speed_change"):
        raise ValueError("pitch/speed augmentation is not implemented (Training Plan Table 7 item 1).")
    return cfg


def gate_problems(cfg: Config) -> list[str]:
    g = cfg["gate"]
    problems = []
    for key in ("tier_criteria_version", "labeling_protocol_version", "label_file_version",
                "target_macro_f1", "min_high_recall", "clarity_scale_version",
                "filler_marking_protocol_version"):
        if g.get(key) in (None, ""):
            problems.append(f"gate.{key} is empty")
    if not g.get("stack_confirmed"):
        problems.append("gate.stack_confirmed is false (G9)")
    if not g.get("storage_approved"):
        problems.append("gate.storage_approved is false (G10)")
    if not cfg["cnn14"].get("checkpoint"):
        problems.append("cnn14.checkpoint is empty (pretrained weights required)")
    if cfg["clarity"].get("scale_min") is None or cfg["clarity"].get("scale_max") is None:
        problems.append("clarity.scale_min / scale_max are empty")
    crit = Path(cfg["paths"].get("criteria_file", ""))
    crit = crit if crit.is_absolute() else Path(cfg["_root"]) / crit
    c = (yaml.safe_load(crit.read_text()) or {}) if crit.is_file() else {}
    if not c.get("version") or not c.get("rules"):
        problems.append(f"tier criteria file has no version or rules ({crit.name}) (G1)")
    if cfg["split"].get("min_test_learners_per_tier") is None:
        problems.append("split.min_test_learners_per_tier is empty")
    return problems


def check_gate(cfg: Config) -> None:
    """In study mode, stop if any gate item is open. In practice mode, print a banner."""
    problems = gate_problems(cfg)
    if cfg["mode"] == "study":
        if problems:
            raise GateError("Readiness gate is not closed:\n  - " + "\n  - ".join(problems))
    else:
        print("=" * 72)
        print("PRACTICE MODE: pipeline check only. These numbers are NOT results and")
        print("must never be reported in the thesis (Training Plan, Section 2).")
        if problems:
            print(f"Open gate items: {len(problems)} (run `python -m readsmart gate` to list)")
        print("=" * 72)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def checkpoint_path(cfg, trained_with: str | None = None) -> str | None:
    """CNN14 checkpoint path from config.yaml, resolved relative to the config file.
    trained_with: the checkpoint a saved model was trained with; a different file name is refused,
    because features from other weights would not match the saved heads."""
    ck = cfg["cnn14"].get("checkpoint")
    if trained_with and not ck:
        raise FileNotFoundError("This saved model was trained with the CNN14 checkpoint "
                                f"'{Path(trained_with).name}'. Set cnn14.checkpoint in config.yaml to that file.")
    if not trained_with and ck is None:
        return None
    if ck is None:
        return None
    p = Path(ck)
    p = p if p.is_absolute() else Path(cfg["_root"]) / p
    if not p.is_file():
        raise FileNotFoundError(f"CNN14 checkpoint not found: {p}. See the README, section 'Get the CNN14 weights'.")
    if trained_with and Path(trained_with).name != p.name:
        raise ValueError(f"Saved model was trained with '{Path(trained_with).name}' but config.yaml points to '{p.name}'.")
    return str(p)
