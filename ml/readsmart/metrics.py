"""Accuracy, per-tier precision/recall/F1, macro and weighted averages, confusion matrix (Section 8.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support


def evaluate(y_true, y_pred, tiers: list[str]) -> dict:
    labels = list(range(len(tiers)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred), "n": int(len(y_true))}
    for i, t in enumerate(tiers):
        out.update({f"precision_{t}": p[i], f"recall_{t}": r[i], f"f1_{t}": f[i], f"support_{t}": int(s[i])})
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out.update({f"precision_{avg}": pa, f"recall_{avg}": ra, f"f1_{avg}": fa})
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["confusion_matrix"] = cm.tolist()
    # errors between neighboring tiers vs. far apart (Low <-> High), Section 8.1
    off = np.abs(np.subtract.outer(labels, labels))
    out["errors_adjacent"] = int(cm[off == 1].sum())
    out["errors_far"] = int(cm[off >= 2].sum())
    return out


def report_table(m: dict, tiers: list[str]) -> pd.DataFrame:
    """Training Plan Table 5 layout."""
    rows = [[t, m[f"precision_{t}"], m[f"recall_{t}"], m[f"f1_{t}"], m[f"support_{t}"]] for t in tiers]
    tot = sum(m[f"support_{t}"] for t in tiers)
    rows += [["Macro average", m["precision_macro"], m["recall_macro"], m["f1_macro"], tot],
             ["Weighted average", m["precision_weighted"], m["recall_weighted"], m["f1_weighted"], tot],
             ["Accuracy", None, None, m["accuracy"], tot]]
    return pd.DataFrame(rows, columns=["Tier", "Precision", "Recall", "F1-score", "Support"]).round(4)


def plot_confusion(m: dict, tiers: list[str], title: str, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=200)
    ax.imshow(cm, cmap="Blues")
    for i in range(len(tiers)):
        for j in range(len(tiers)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks(range(len(tiers)), tiers)
    ax.set_yticks(range(len(tiers)), tiers)
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Reference tier")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def rater_agreement(a, b, tiers: list[str]) -> dict:
    """Weighted Cohen's kappa for the double-labeled subset (Section 3.3).
    [DECIDE] linear vs quadratic weights; [VERIFY] citation before using in Chapter III."""
    idx = {t: i for i, t in enumerate(tiers)}
    ya, yb = [idx[t] for t in a], [idx[t] for t in b]
    return {"n": len(ya),
            "percent_agreement": float(np.mean(np.array(ya) == np.array(yb))),
            "kappa_unweighted": cohen_kappa_score(ya, yb),
            "kappa_linear": cohen_kappa_score(ya, yb, weights="linear"),
            "kappa_quadratic": cohen_kappa_score(ya, yb, weights="quadratic")}
