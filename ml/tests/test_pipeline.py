"""Fast checks of the parts the panel is most likely to question: leakage, metrics, CNN14 shape."""
import numpy as np
import pandas as pd
import pytest
import torch

from readsmart import audio as A
from readsmart.cnn14 import Cnn14
from readsmart.metrics import evaluate
from readsmart.splits import LeakageError, check_no_leakage, learner_table

TIERS = ["Low", "Moderate", "High"]


def test_leakage_check_catches_shared_learner():
    with pytest.raises(LeakageError):
        check_no_leakage(["L1", "L2"], ["L2", "L3"])
    with pytest.raises(LeakageError):
        check_no_leakage(["L1"], ["L2"], test_learners=["L1"])
    with pytest.raises(LeakageError):
        check_no_leakage(["L1"], ["L2"], test_learners=["L2"])  # test learner used for validation
    check_no_leakage(["L1"], ["L2"], test_learners=["L2"], eval_is_test=True)


def test_learner_table_one_row_per_learner():
    rec = pd.DataFrame({"learner_code": ["A", "A", "B"], "reference_tier": ["Low", "Low", "High"]})
    lt = learner_table(rec, TIERS)
    assert sorted(lt.learner_code) == ["A", "B"]


def test_metrics_and_far_errors():
    y = np.array([0, 0, 1, 2, 2])
    p = np.array([0, 1, 1, 0, 2])
    m = evaluate(y, p, TIERS)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["errors_adjacent"] == 1 and m["errors_far"] == 1
    assert np.array(m["confusion_matrix"]).sum() == 5


def test_segmentation_lengths():
    sr = 16000
    y = np.random.default_rng(0).standard_normal(sr * 12).astype(np.float32) * 0.1
    segs = A.segment(y, sr, 5, 0.25, 30, 0.5)
    assert all(len(c) == 5 * sr for _, _, c, _, _ in segs)
    assert segs[-1][1] + 5 >= 12 - 1e-6  # tail covered


def test_quality_check_flags_short_and_quiet():
    sr = 16000
    qc = {"min_duration_s": 10, "min_rms_dbfs": -45, "max_clipped_fraction": 0.001}
    short = np.random.default_rng(0).standard_normal(sr * 3).astype(np.float32) * 0.1
    assert not A.quality_check(short, sr, qc, 30).passed
    quiet = np.random.default_rng(0).standard_normal(sr * 20).astype(np.float32) * 1e-4
    assert "too quiet" in A.quality_check(quiet, sr, qc, 30).reasons


def test_cnn14_matches_panns_layout():
    m = Cnn14().eval()
    keys = set(m.state_dict())
    # parameter names used in the PANNs checkpoint (load uses strict=True, so any mismatch fails loudly)
    for k in ["spectrogram_extractor.stft.conv_real.weight", "logmel_extractor.melW", "bn0.running_mean",
              "conv_block6.conv2.weight", "fc1.weight", "fc_audioset.weight"]:
        assert k in keys
    with torch.no_grad():
        e = m(torch.zeros(2, 16000 * 3))
    assert tuple(e.shape) == (2, 2048)


# --- speech analyzer and tier step ---
from readsmart.analyzer import events_from_probs, frame_labels
from readsmart.analyzer_eval import match_events
from readsmart.tiers import CriteriaError, assign_tier, load_criteria


def test_frame_labels_and_events_round_trip():
    lab = frame_labels(50, [(0.8, 1.2)])  # 80 ms frames
    assert lab[10:15].sum() == 5 and lab.sum() == 5
    ev = events_from_probs(lab, 0.5, 0.16, 0.16)
    assert len(ev) == 1 and abs(ev[0][0] - 0.8) < 0.09 and abs(ev[0][1] - 1.2) < 0.09


def test_match_events():
    tp, fp, fn = match_events([(1.0, 1.5), (3.0, 3.2)], [(1.1, 1.6), (5.0, 5.4)])
    assert (tp, fp, fn) == (1, 1, 1)


def test_tier_step_without_criteria_gives_no_tier(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("version: null\napproved_by: null\ndefault_tier: null\nrules: []\n")
    assert load_criteria(p, TIERS) is None
    assert assign_tier({"clarity_score": 1}, None)["status"] == "criteria not set"


def test_tier_rules_first_match_wins(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("""version: t
approved_by: test
default_tier: Low
rules:
  - tier: High
    any: [{metric: clarity_score, op: "<=", value: 1}]
  - tier: Moderate
    all: [{metric: clarity_score, op: "<=", value: 2}, {metric: filler_count, op: ">=", value: 1}]
""")
    c = load_criteria(p, TIERS)
    assert assign_tier({"clarity_score": 1, "filler_count": 0}, c)["tier"] == "High"
    assert assign_tier({"clarity_score": 2, "filler_count": 1}, c)["tier"] == "Moderate"
    assert assign_tier({"clarity_score": 2, "filler_count": 0}, c)["tier"] == "Low"
    p.write_text("version: t\napproved_by: x\ndefault_tier: Low\nrules:\n  - tier: High\n    any: [{metric: wcpm, op: '<', value: 1}]\n")
    with pytest.raises(CriteriaError):
        load_criteria(p, TIERS)
