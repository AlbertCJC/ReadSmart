"""Learner-level splits: a locked test set, then StratifiedGroupKFold on the rest (Training Plan, Section 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import usable_recordings, work


class LeakageError(RuntimeError):
    pass


def learner_table(rec: pd.DataFrame, tiers: list[str]) -> pd.DataFrame:
    """One row per learner with one tier. If a learner has recordings with different tiers,
    the highest-priority tier is used for stratification only and a warning is printed."""
    order = {t: i for i, t in enumerate(tiers)}
    rows = []
    for code, g in rec.groupby("learner_code"):
        ts = set(g.reference_tier)
        if len(ts) > 1:
            print(f"WARNING: learner {code} has recordings with different tiers {ts} (Table 7 item 4).")
        rows.append({"learner_code": code, "reference_tier": max(ts, key=order.get)})
    return pd.DataFrame(rows)


def lock_test_set(cfg, force: bool = False) -> pd.DataFrame:
    p = work(cfg, "splits", "test_learners.csv")
    if p.exists() and not force:
        raise FileExistsError(f"Test set is already locked at {p}. It must not be redrawn. "
                              "Use --force only if the label file version changed (and record why).")
    rec = usable_recordings(cfg)
    lt = learner_table(rec, cfg["tiers"])
    n_splits = max(2, int(round(1 / cfg["split"]["test_share"])))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    _, test_idx = next(sgkf.split(lt, lt.reference_tier, groups=lt.learner_code))
    test = lt.iloc[test_idx].sort_values("learner_code")
    test.to_csv(p, index=False)
    counts = test.reference_tier.value_counts().reindex(cfg["tiers"]).fillna(0).astype(int)
    print(f"Locked test set: {len(test)} of {len(lt)} learners -> {p}\n{counts.to_string()}")
    floor = cfg["split"].get("min_test_learners_per_tier")
    if floor is None:
        print("NOTE: split.min_test_learners_per_tier is [DECIDE]; not checked.")
    elif (counts < floor).any():
        print(f"WARNING: some tiers have fewer than {floor} test learners: {counts.to_dict()}")
    return test


def make_folds(cfg, force: bool = False) -> pd.DataFrame:
    p = work(cfg, "splits", "folds.csv")
    if p.exists() and not force:
        print(f"Using existing fold file {p}")
        return pd.read_csv(p, dtype={"learner_code": str})
    test = set(pd.read_csv(work(cfg, "splits", "test_learners.csv"), dtype=str).learner_code)
    rec = usable_recordings(cfg)
    dev = learner_table(rec[~rec.learner_code.isin(test)], cfg["tiers"])
    smallest = dev.reference_tier.value_counts().min()
    k = cfg["split"]["n_folds"]
    if smallest < k:
        print(f"NOTE: smallest development tier has {smallest} learners; using {smallest} folds instead of {k} "
              "(report this in the results).")
        k = int(smallest)
    if k < 2:
        raise ValueError("Fewer than 2 learners in some development tier: cross-validation is not possible.")
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg["seed"])
    dev["fold"] = -1
    for f, (_, va) in enumerate(sgkf.split(dev, dev.reference_tier, groups=dev.learner_code)):
        dev.loc[dev.index[va], "fold"] = f
    dev.to_csv(p, index=False)
    print(f"Saved {k}-fold assignment for {len(dev)} development learners -> {p}")
    print(pd.crosstab(dev.fold, dev.reference_tier).to_string())
    return dev


def load_splits(cfg):
    test = pd.read_csv(work(cfg, "splits", "test_learners.csv"), dtype=str)
    folds = pd.read_csv(work(cfg, "splits", "folds.csv"), dtype={"learner_code": str})
    return test, folds


def check_no_leakage(train_learners, eval_learners, test_learners=None, eval_is_test=False):
    """Leakage checklist items 1-2: a learner sits in exactly one of train / validation / test."""
    tr, ev = set(train_learners), set(eval_learners)
    if tr & ev:
        raise LeakageError(f"learners in both training and evaluation: {sorted(tr & ev)[:5]}")
    if test_learners is not None:
        te = set(test_learners)
        if tr & te:
            raise LeakageError(f"test learners in training data: {sorted(tr & te)[:5]}")
        if not eval_is_test and ev & te:
            raise LeakageError(f"test learners used for validation: {sorted(ev & te)[:5]}")


def fold_learners(folds: pd.DataFrame, f: int):
    return (folds.loc[folds.fold != f, "learner_code"].to_numpy(),
            folds.loc[folds.fold == f, "learner_code"].to_numpy())


def all_dev_learners(folds: pd.DataFrame) -> np.ndarray:
    return folds.learner_code.to_numpy()
