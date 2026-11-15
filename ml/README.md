# Read Smart 2.0: training and evaluation pipeline

Code for the offline "Model Development and Evaluation" block of Figure 3.19. It implements
the Model Training Plan (Nov 15, 2026) so that training can start
as soon as real, consented, labeled recordings exist.

**Status: proposed, pipeline-tested only.** The stack (Python, PyTorch, PANNs CNN14,
scikit-learn) is still a proposal (gate item G9). There is no real data, no tier criteria and no
pretrained checkpoint in this folder. The only runs so far used synthetic tone sequences, and
those numbers are **not results**. They only show that the code runs from start to finish.

Framing that applies to every output: the model assigns a **study-defined instructional
prioritization tier** (Low, Moderate, High). It does not diagnose anything, does not measure
words correct per minute or word accuracy, and does not report error types. Teachers keep final
authority.

---

## 0. Main design: one speech-metrics model + separate tier step

As of the later Nov 15 decision (see `../docs/DESIGN_speech_metrics.md`), the main system is:

1. **Speech analyzer (one model).** Audio in, speech metrics out: pauses, speaking time, syllable-like
   peaks, speech and articulation rate (from the signal), **filler count and time** (filler head,
   trained on marked "uh"/"um"), and a **clarity score** (clarity head, trained on rater scores).
2. **Tier step.** Applies the rules in `criteria/tier_criteria.yaml` (the tier criteria, Objective 4)
   to the metrics and returns Low, Moderate or High. The file is empty, so for now every result says
   "criteria not set". The code never sets a threshold.
3. **Output:** metrics + tier + criteria version.

```bash
python -m readsmart qc
python -m readsmart split                 # same locked test set and folds as below
python -m readsmart import-audacity --labels-dir data/filler_labels   # Audacity label exports
python -m readsmart frames                # CNN14 frame features
python -m readsmart timing-check          # automatic vs hand-timed pauses/syllables (dev learners)
python -m readsmart criteria-check
python -m readsmart analyzer-cv
python -m readsmart analyzer-final --version-label v1.0     # test set used once
python -m readsmart analyze --run runs/<analyzer final run> --wav some.wav
```

Extra labels it needs (templates in `templates/`): `clarity_score` and `fillers_marked` columns in
`labels.csv`, `data/filler_marks.csv` (start/end of each filler, in original-file seconds) and
`data/timing_checks.csv`. Filler marks are **required**. Try it on fake audio with
`python scripts/make_synthetic_data.py --out synthetic` then
`python -m readsmart practice-analyzer --config synthetic/config.yaml` (about 2 minutes). The
synthetic folder has its own made-up `criteria_synthetic.yaml` so the tier step can be exercised.
Those rules are invented and are not tier criteria.

Sections 1 to 9 below describe the earlier direct-tier classifiers (M0 to M5). They share the
quality check, splits and safeguards, and remain available as comparison baselines.

## 1. What the pipeline does

| Step | Command | Training Plan |
|---|---|---|
| Quality check (min duration, min loudness, clipping), then mono, 16 kHz, trim silence, loudness normalization | `qc` | 3.1, Table 3 |
| Cut 3, 5 and 10 s clips with overlap, drop mostly-silent clips, log drops per recording | `clips` | Table 3 |
| Optional M1 simple acoustic measures (pauses, onset rate, pitch, loudness spread) | `handcrafted` | Table 4, Table 7 item 5 |
| 64-band log-mel (32 ms window, 10 ms hop, 50 to 8,000 Hz) into frozen CNN14, cache 2,048-d embeddings, plus augmented copies for training only | `embed` | 5, 7 |
| Lock ~20% of learners as the test set (stratified, grouped by learner), then StratifiedGroupKFold on the rest | `split` | 4 |
| Dataset inventory (Table 2) | `inventory` | 3.2 |
| Weighted Cohen's kappa for the double-labeled subset | `kappa` | 3.3 |
| Cross-validate M0 to M5 with the grids from Section 6 | `cv` | 6 |
| Choose the model by CV macro F1, High-recall floor, simpler-if-within-1-SD | `select` | 8.2 |
| Retrain on all development learners, test once on the locked test set, save Table 5, confusion matrices, model file and MODEL_VERSION card | `final` | 8.3, 9 |
| Classify one WAV with a saved model | `predict` | Module 7 |
| Sample log-mel figure for Chapter III | `figure` | 8.4 |

Models:

| ID | Model | Input |
|---|---|---|
| M0 | Majority class | anything (floor) |
| M1 | Logistic regression, balanced class weights | 12 hand-crafted measures per recording (optional) |
| M2 | **Logistic regression, balanced class weights (main proposed model)** | CNN14 embeddings |
| M3 | SVM (linear or RBF), balanced class weights | CNN14 embeddings |
| M4 | Small head 2048 to 256 to 3, dropout, weighted cross-entropy | CNN14 embeddings |
| M5 | Small CNN from scratch (4 conv blocks, global pooling, 3 outputs) | the same log-mel spectrograms |

M6 (fine-tuning CNN14) is not implemented. The plan runs it only if M2 to M4 underfit and there is enough data.

Every M2 to M4 configuration is tried with both clip-to-recording aggregations from Section 5:
`mean_embedding` (average clip embeddings, then classify) and `mean_proba` (classify clips, average probabilities).

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests                              # 10 quick checks
```

Download the PANNs CNN14 16 kHz checkpoint (MIT license) yourself and set its path in `config.yaml`:

- File: `Cnn14_16k_mAP=0.438.pth` from the PANNs Zenodo record (zenodo.org/record/3987831), linked from github.com/qiuqiangkong/audioset_tagging_cnn
- Put it in `checkpoints/` and set `cnn14.checkpoint: checkpoints/Cnn14_16k_mAP=0.438.pth`

The checkpoint is loaded with `strict=True`, so a wrong file fails loudly instead of loading partly.
Without a checkpoint the code uses random CNN14 weights, which is allowed only in practice mode.

## 3. Try it on synthetic audio first (5 to 20 minutes on a laptop CPU)

```bash
python scripts/make_synthetic_data.py --out synthetic
python -m readsmart practice-all --config synthetic/config.yaml
```

This writes everything under `synthetic/` (its own `data/`, `work/`, `runs/`), so it can never mix
with study data. To make it faster, set `models.M5.enabled: false` in `synthetic/config.yaml`,
or run M5 separately with `python -m readsmart cv --models M5 --config synthetic/config.yaml`.
`--shuffle-labels` makes a copy with random labels; scores should then drop to chance, which is
a quick way to check that nothing leaks.

## 4. Running it on the study data

1. **Close the readiness gate.** `python -m readsmart gate` lists what is still open. Fill in the
   `gate:` section of `config.yaml` and set `mode: study`. In study mode every command refuses to
   run until all gate items are filled, a checkpoint is set, and the label file version matches.
2. **Put the data in place** (never in Git; see `.gitignore`):
   - `data/raw/` holds the WAV files (named by anonymized learner code only)
   - `data/labels.csv` follows `templates/labels_template.csv`: one row per recording, failed
     attempts kept with an empty tier, one frozen `label_version` for the whole file
   - `data/double_labels.csv` follows `templates/double_labels_template.csv`
   - optional `data/excluded_clips.csv` (`templates/excluded_clips_template.csv`) for clips with
     the teacher's voice, which the code cannot detect by itself
   - keep the session log (`templates/session_log_template.csv`) next to the data
3. **Run the steps in order** (Training Plan Table 6):

```bash
python -m readsmart qc
python -m readsmart clips
python -m readsmart handcrafted         # only if M1 is kept
python -m readsmart embed
python -m readsmart split               # locks the test set; refuses to redraw it later
python -m readsmart inventory           # Table 2 for Chapter IV
python -m readsmart kappa
python -m readsmart cv                  # or --models M0,M2 first, then M3,M4, then M5
python -m readsmart select              # writes work/decision.json; record it before the next step
python -m readsmart final --version-label v1.0
python -m readsmart predict --run runs/<final run folder> --wav some.wav
```

`final` writes `work/splits/TEST_SET_OPENED.json`. A second final test is refused unless you give
`--force-reason "..."`, which goes into the run log.

## 5. What each run saves

Every run gets its own folder in `runs/` with the exact config used, `environment.txt` (Python and
library versions) and, in practice mode, a `PRACTICE_NOT_RESULTS.txt` marker. `runs/run_log.csv`
gets one row per run (date, mode, label version, criteria version, fold file, clip lengths,
checkpoint, seed, headline numbers).

- `cv_folds.csv`: every metric for every model, setting and fold
- `cv_summary.csv`: mean and SD across folds, sorted by macro F1
- `final_test`: `table5_<role>_<model>.csv` (Training Plan Table 5 layout), `confusion_<...>.png`,
  `predictions_<...>.csv`, `test_results.json`, `model.joblib` or `model.pt`, and
  `model_card.json` with the MODEL_VERSION fields from Chapter III Table 3.19 (`is_active` stays
  false; Module 9 sets it)

Metrics reported everywhere: accuracy; precision, recall and F1 per tier; macro and weighted
averages; the 3x3 confusion matrix; and how many errors were between neighboring tiers versus Low
against High.

## 6. Safeguards built in

- **Learner-level splits only.** Test learners and fold learners are checked for overlap before
  every fit (`check_no_leakage`); any overlap stops the run.
- **Augmented copies never reach evaluation.** Validation and test views use original clips only.
- **Scaling is fitted on training data only** (inside each model).
- **Early stopping (M4, M5) uses a slice of the training learners**, not the evaluation fold, so
  the CV score is not used to pick the stopping epoch. The final retrain uses the median best
  epoch from CV. This is slightly stricter than the plan, which says "early stopping on the
  validation fold".
- **The test set is locked once** and can only be used once without a logged reason.
- **Class weights, not oversampling** (Table 7 item 2). **No pitch or speed augmentation**
  (Table 7 item 1); setting either to true in the config is refused.
- **Folds shrink automatically** if the smallest development tier has fewer than 5 learners, and
  the run says so.

## 7. Placeholders left for the team (not invented)

| Config key | Marker | Source |
|---|---|---|
| `gate.tier_criteria_version` | TO BE FILLED | G1, Objective 4 |
| `gate.labeling_protocol_version` | TO BE FILLED | G2 |
| `gate.label_file_version` | TO BE FILLED | Section 3.3 |
| `gate.target_macro_f1`, `gate.min_high_recall` | TO BE FILLED | G8, Validation Board S1 |
| `gate.stack_confirmed` | DECIDE | G9 |
| `gate.storage_approved` | TO BE FILLED | G10 |
| `cnn14.checkpoint` | download | Section 5 |
| `audio.qc.*`, `audio.trim_top_db` | DECIDE | set from a few real recordings |
| `augment.noise_dir` | TO BE FILLED | classroom noise samples; white noise is a stand-in |
| `augment.pitch_shift`, `speed_change` | DECIDE | Table 7 item 1 (off) |
| `imbalance` | DECIDE | Table 7 item 2 (class weights) |
| `spectrogram.*` | DECIDE | Table 7 item 3 (64 mels, 32 ms) |
| `split.test_share`, `min_test_learners_per_tier` | DECIDE | Section 4, 3.2 |
| `split.prediction_unit` | DECIDE | Table 7 item 4 |
| `models.M1.enabled` | DECIDE | Table 7 item 5 |
| Kappa weighting (linear or quadratic) | DECIDE, VERIFY citation | Section 3.3; `kappa` prints both |

Table 7 items 6 (database) and 8 (cloud compute for learner audio) are not code settings, but item
8 matters here: do not run this on a cloud notebook with learner audio until G10 says it is allowed.

## 8. Synthetic test run (Nov 15, 2026), pipeline check only

Ran on 36 synthetic "learners" (17 Low, 12 Moderate, 7 High labels assigned to tone sequences
that differ in pause and noise level), random CNN14 weights, CPU only. Every step ran: 2 files
failed the quality check as intended, the test set locked 7 learners (3/2/2), 5 grouped folds
were built, 102 model configurations were cross-validated (M5 with one reduced setting to save CPU time), a model was selected and the final test,
model card and `predict` all worked. **None of these numbers say anything about real learners
or about how well Read Smart 2.0 will work.** See `SYNTHETIC_TEST_LOG.md` for the exact output.

## 9. Folder layout

```
ml/
  config.yaml            all settings and placeholders
  requirements.txt       proposed stack, versions tested
  readsmart/             the package (python -m readsmart <command>)
    audio.py             load, quality check, trim, normalize, segment, M1 measures
    cnn14.py             PANNs CNN14 layers (MIT, copied with license) + log-mel front end
    embed.py             embedding cache with augmented copies
    splits.py            locked test set, grouped folds, leakage check
    models.py            M0 to M5
    experiments.py       CV, selection, final test, model card
    metrics.py           metrics, Table 5, confusion matrix, kappa
    predict.py           classify one file
    timing.py            pauses, syllable peaks, rates (speech analyzer)
    analyzer.py          speech analyzer: filler head, clarity head, save/load
    analyzer_eval.py     timing-check, analyzer-cv, analyzer-final, analyze
    tiers.py             tier step: reads and applies criteria/tier_criteria.yaml
  criteria/tier_criteria.yaml   empty tier criteria (Objective 4)
  scripts/make_synthetic_data.py
  templates/             labels, double labels, session log, excluded clips
  tests/                 quick checks
```

Never commit `data/`, `work/`, `runs/`, checkpoints or anything with learner audio (Training Plan Section 9).
