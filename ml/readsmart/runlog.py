"""Run folders and the run log (Training Plan Section 9)."""
from __future__ import annotations

import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def new_run_dir(cfg, kind: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = cfg.path("runs_dir") / f"{stamp}_{cfg['mode']}_{kind}"
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy(cfg["_path"], run / "config_used.yaml")
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    except Exception:  # pragma: no cover
        freeze = ""
    (run / "environment.txt").write_text(f"python {platform.python_version()} on {platform.platform()}\n{freeze}")
    if cfg["mode"] == "practice":
        (run / "PRACTICE_NOT_RESULTS.txt").write_text(
            "Practice-mode run on practice or synthetic audio. Never report these numbers.\n")
    return run


def append_run_log(cfg, run: Path, kind: str, extra: dict) -> None:
    log = cfg.path("runs_dir") / "run_log.csv"
    work = cfg.path("work_dir")
    row = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "mode": cfg["mode"], "kind": kind, "run_dir": run.name,
        "label_file_version": cfg["gate"]["label_file_version"],
        "tier_criteria_version": cfg["gate"]["tier_criteria_version"],
        "fold_file": str(work / "splits" / "folds.csv"),
        "clip_lengths_s": json.dumps(cfg["audio"]["clip_lengths_s"]),
        "cnn14_checkpoint": cfg["cnn14"]["checkpoint"], "seed": cfg["seed"],
        **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in extra.items()},
    }
    df = pd.DataFrame([row])
    if log.exists():
        df = pd.concat([pd.read_csv(log), df], ignore_index=True)
    df.to_csv(log, index=False)
