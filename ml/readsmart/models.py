"""Models M0-M5 (Training Plan Table 4). Every model outputs probabilities over the 3 tiers.

Feature scaling sits inside each model and is fitted on training data only
(leakage checklist item 3). Class imbalance is handled with class weights only.
"""
from __future__ import annotations

import copy
import itertools
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

from .cnn14 import LogMel, spec_mask

N_CLASSES = 3
# scikit-learn 1.9 deprecates SVC(probability=True); it still works up to 1.10. Revisit if the stack is upgraded.
warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated", category=FutureWarning)
COMPLEXITY = {"M0": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5}


def param_grid(name: str, mcfg: dict) -> list[dict]:
    if name == "M0":
        return [{}]
    if name in ("M1", "M2"):
        return [{"C": c} for c in mcfg["C"]]
    if name == "M3":
        return [{"C": c, "kernel": k} for c, k in itertools.product(mcfg["C"], mcfg["kernel"])]
    if name in ("M4", "M5"):
        return [{"lr": lr, "dropout": d} for lr, d in itertools.product(mcfg["lr"], mcfg["dropout"])]
    raise ValueError(name)


def full_proba(model, X, n=N_CLASSES):
    """predict_proba with a column for every tier, even if a tier was absent from training."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), n))
    out[:, model.classes_] = p
    return out


class SkModel:
    """M0-M3 wrapper with the same fit/predict_proba interface as the torch models."""

    def __init__(self, name, params, seed):
        if name == "M0":
            est = DummyClassifier(strategy="most_frequent")
        elif name in ("M1", "M2"):
            est = make_pipeline(StandardScaler(), LogisticRegression(
                C=params["C"], class_weight="balanced", max_iter=5000, random_state=seed))
        elif name == "M3":
            est = make_pipeline(StandardScaler(), SVC(
                C=params["C"], kernel=params["kernel"], class_weight="balanced",
                probability=True, random_state=seed))
        self.est = est

    def fit(self, X, y, X_val=None, y_val=None):
        self.est.fit(X, y)
        self.best_epoch = None
        return self

    @property
    def classes_(self):
        return self.est.classes_

    def predict_proba(self, X):
        return full_proba(self.est, X)


def _class_weights(y):
    present = np.unique(y)
    w = np.ones(N_CLASSES, dtype=np.float32)
    w[present] = compute_class_weight("balanced", classes=present, y=y)
    return torch.tensor(w)


class _TorchTrainer:
    """Adam + class-weighted cross-entropy + early stopping on the validation fold (by macro F1).
    When no validation data is given (final retrain), trains for a fixed number of epochs."""

    def __init__(self, params, mcfg, seed, device="cpu"):
        self.p, self.m, self.seed, self.device = params, mcfg, seed, device

    def _batches(self, n, bs, rng):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            yield idx[s:s + bs]

    def _fit_loop(self, net, Xtr, ytr, Xva, yva, epochs_fixed=None):
        from sklearn.metrics import f1_score
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        opt = torch.optim.Adam(net.parameters(), lr=self.p["lr"], weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(weight=_class_weights(ytr).to(self.device))
        max_ep = epochs_fixed or self.m["max_epochs"]
        best, best_state, best_ep, wait = -1, None, 0, 0
        bs = self.m.get("batch_size", 32)
        for ep in range(1, max_ep + 1):
            net.train()
            for b in self._batches(len(ytr), bs, rng):
                xb = self._to_input(Xtr, b, train=True)
                loss = lossf(net(xb), torch.as_tensor(ytr[b], device=self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epochs_fixed is None and Xva is not None:
                pred = self._predict(net, Xva).argmax(1)
                score = f1_score(yva, pred, average="macro", labels=range(N_CLASSES), zero_division=0)
                if score > best:
                    best, best_state, best_ep, wait = score, copy.deepcopy(net.state_dict()), ep, 0
                else:
                    wait += 1
                    if wait >= self.m["patience"]:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.best_epoch = best_ep if epochs_fixed is None else epochs_fixed
        return net

    @torch.no_grad()
    def _predict(self, net, X):
        net.eval()
        out = []
        for s in range(0, len(X), 64):
            out.append(F.softmax(net(self._to_input(X, np.arange(s, min(s + 64, len(X))), train=False)), 1).cpu())
        return torch.cat(out).numpy()

    def predict_proba(self, X):
        return self._predict(self.net, X)


class HeadModel(_TorchTrainer):
    """M4: small neural head on CNN14 embeddings: 2048 -> hidden -> 3, dropout."""

    def _to_input(self, X, idx, train):
        return torch.as_tensor((X[idx] - self.mu) / self.sd, dtype=torch.float32, device=self.device)

    def fit(self, X, y, X_val=None, y_val=None, epochs=None):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6  # fitted on training data only
        net = nn.Sequential(nn.Dropout(self.p["dropout"]), nn.Linear(X.shape[1], self.m["hidden"]), nn.ReLU(),
                            nn.Dropout(self.p["dropout"]), nn.Linear(self.m["hidden"], N_CLASSES)).to(self.device)
        self.net = self._fit_loop(net, X, y, X_val, y_val, epochs)
        return self


class SmallCNN(nn.Module):
    """M5: about four conv blocks on log-mel, global pooling, 3 outputs; trained from scratch."""

    def __init__(self, n_mels, dropout):
        super().__init__()
        chans = [1, 32, 64, 128, 256]
        blocks = []
        for a, b in zip(chans[:-1], chans[1:]):
            blocks += [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(), nn.AvgPool2d(2)]
        self.bn0 = nn.BatchNorm2d(n_mels)
        self.features = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(chans[-1], N_CLASSES)

    def forward(self, logmel):  # (B, 1, T, mel)
        x = self.bn0(logmel.transpose(1, 3)).transpose(1, 3)
        x = self.features(x)
        x = x.mean(dim=3)
        x = x.max(dim=2)[0] + x.mean(dim=2)
        return self.fc(self.drop(x))


class ScratchCNNModel(_TorchTrainer):
    """M5 trainer. X is an array of waveform clips (N, samples). Training clips get waveform
    augmentation (gain, noise) and SpecAugment masking; evaluation clips get none."""

    def __init__(self, params, mcfg, seed, device, spec_cfg, sr, augmenter=None, specaug=None):
        super().__init__(params, mcfg, seed, device)
        self.front = LogMel(sr, spec_cfg["n_fft"], spec_cfg["hop_length"], spec_cfg["n_mels"],
                            spec_cfg["fmin"], spec_cfg["fmax"]).to(device).eval()
        self.n_mels, self.augmenter, self.specaug = spec_cfg["n_mels"], augmenter, specaug
        self.gen = torch.Generator().manual_seed(seed)

    def _to_input(self, X, idx, train):
        clips = X[idx]
        if train and self.augmenter is not None:
            clips = np.stack([self.augmenter(c) for c in clips])
        with torch.no_grad():
            x = self.front(torch.as_tensor(clips, device=self.device))
            if train and self.specaug:
                s = self.specaug
                x = spec_mask(x, s["time_masks"], s["time_width"], s["freq_masks"], s["freq_width"], self.gen)
        return x

    def fit(self, X, y, X_val=None, y_val=None, epochs=None):
        net = SmallCNN(self.n_mels, self.p["dropout"]).to(self.device)
        self.net = self._fit_loop(net, X, y, X_val, y_val, epochs)
        return self
