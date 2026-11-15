# Read Smart 2.0

*Read Smart 2.0: An Intelligent Oral Reading Evaluator and Speech Classifier for Early Childhood Intervention* (undergraduate thesis project).

Read Smart 2.0 records a Grade 4 learner reading a short English passage aloud, converts the
audio to a spectrogram, analyzes it with a CNN-based speech analyzer, and assigns a
**Low, Moderate or High** tier that teachers may use when deciding which learners to prioritize
for existing support such as DepEd's ARAL Program.

It is a **supplementary, non-diagnostic decision-support tool**. It does not diagnose any speech,
language or learning disorder, does not measure words correct per minute or word accuracy, and
does not replace Phil-IRI or teacher judgment. The tiers are study-defined instructional
prioritization categories, not clinical categories.

## Status

Prototype in progress. The stack is a proposal and has not been confirmed by the team or adviser.
No learner data, tier criteria or trained model exist yet. The code has only been run on
synthetic audio, and those runs are not results.

## Repository layout

| Folder | Contents |
|---|---|
| `ml/` | Speech analyzer (CNN14 features, filler and clarity heads, timing measures), tier step, training and evaluation pipeline. Start with [`ml/README.md`](ml/README.md) |
| `ml/criteria/` | Tier criteria file (Objective 4). Empty until the researchers write it with the adviser |
| `ml/templates/` | Label, filler-mark, timing-check and session-log templates (anonymized learner codes only) |
| `docs/` | Design notes: [`DESIGN_speech_metrics.md`](docs/DESIGN_speech_metrics.md) |

The teacher-facing web app (recording, results screen, administration) will be added in its own
folder once the stack is confirmed.

## Contents

1. [What you need](#1-what-you-need)
2. [Set up](#2-set-up)
3. [Get the CNN14 weights](#3-get-the-cnn14-weights)
4. [Check that everything works](#4-check-that-everything-works)
5. [Use it: analyze a recording](#5-use-it-analyze-a-recording)
6. [Use a saved model without training](#6-use-a-saved-model-without-training)
7. [Train the model](#7-train-the-model)
8. [Write the tier criteria](#8-write-the-tier-criteria)
9. [Troubleshooting](#9-troubleshooting)

All commands run from the `ml/` folder. More detail on each command is in [`ml/README.md`](ml/README.md).

## 1. What you need

- Python 3.10 or newer (tested on 3.13) and Git
- About 3 GB of free disk space (PyTorch plus the CNN14 weights)
- A normal laptop CPU is enough for everything except the optional from-scratch CNN baseline (M5), which wants a GPU
- Recordings as **WAV** files. Phone formats such as `.m4a` or `.mp3` must be converted to WAV first (Audacity: File > Export > WAV). Any sample rate works; the code converts to mono 16 kHz

## 2. Set up

```bash
git clone https://github.com/AlbertCJC/ReadSmart.git
cd ReadSmart/ml

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows (Command Prompt or PowerShell)

pip install -r requirements.txt
python -m pytest -q tests          # should say "10 passed"
```

## 3. Get the CNN14 weights

The analyzer uses PANNs CNN14 (16 kHz), a network already trained on general audio, as its
feature extractor. Its weights are not in this repository (large file). Download once:

- File name: `Cnn14_16k_mAP=0.438.pth` (a few hundred MB), MIT license
- From the PANNs Zenodo record: <https://zenodo.org/record/3987831>
  (linked from <https://github.com/qiuqiangkong/audioset_tagging_cnn>)

```bash
mkdir checkpoints
# macOS / Linux
curl -L -o "checkpoints/Cnn14_16k_mAP=0.438.pth" "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
# Windows: download the file in a browser and move it into ml\checkpoints\
```

Then open `ml/config.yaml` and set:

```yaml
cnn14:
  checkpoint: checkpoints/Cnn14_16k_mAP=0.438.pth
```

The path is relative to `config.yaml`. A wrong or damaged file stops with an error instead of
loading partly. Loading the real weights has not been tested yet (the download site was not
reachable where the code was written), so tell the team if `frames` fails on the first run.

**These weights alone are not a Read Smart model.** They only turn audio into features. The
parts that produce filler counts, the clarity score and the tier must be trained on the team's
own labeled recordings (section 7) or loaded from a saved model (section 6).

## 4. Check that everything works

This makes fake audio (tone sequences, not speech), trains a throwaway model on it and tests it.
It takes about 2 minutes and needs no real data. If `cnn14.checkpoint` is still empty it uses
random weights, which is fine for this check only.

```bash
python scripts/make_synthetic_data.py --out synthetic
python -m readsmart practice-analyzer --config synthetic/config.yaml
```

Everything it writes stays in `ml/synthetic/`, which Git ignores. **Numbers from this run are not
results** and must never be reported.

## 5. Use it: analyze a recording

You need a saved model folder (from section 6 or 7). Then:

```bash
python -m readsmart analyze --model <model folder> --wav path/to/recording.wav
```

To try it right after section 4, use the throwaway model. The run folder name starts with the date
and time and ends in `_practice_analyzer_final_test` (look in `synthetic/runs/`):

```bash
python -m readsmart analyze --config synthetic/config.yaml \
    --model synthetic/runs/<date-time>_practice_analyzer_final_test \
    --wav synthetic/data/raw/SYN010_take1.wav
```

### Reading the output

```json
{
  "quality_status": "passed",
  "metrics": { "pause_count": 10, "fillers_per_min": 0.0, "clarity_score": 3.2, "...": "..." },
  "filler_events_s": [[3.44, 3.92]],
  "tier": "Moderate",
  "criteria_version": "criteria v1",
  "status": "ok",
  "matched_rule": 1,
  "model_version": "v1.0"
}
```

| Field | Meaning |
|---|---|
| `quality_status` | `passed`, or `failed` with the reason (too short, too quiet, clipping). A failed recording gets no metrics: record again |
| `duration_s`, `speaking_time_s`, `speaking_ratio` | Length after trimming silence at both ends, time not spent in pauses, and their ratio |
| `pause_count`, `pauses_per_min`, `total_pause_s`, `mean_pause_s`, `longest_pause_s` | Silences of at least `timing.min_pause_s` between speech |
| `syllable_count`, `speech_rate_syl_per_s`, `articulation_rate_syl_per_s` | Syllable-like loudness peaks; per second of the whole recording, and per second of speaking time without fillers. A rough rate, **not** words per minute |
| `filler_count`, `fillers_per_min`, `filler_time_s`, `filler_events_s` | Detected "uh"/"um" events and their start and end times in seconds |
| `clarity_score` | Estimated clarity on the raters' scale |
| `tier` | Low, Moderate or High from the tier criteria. `null` with `status: "criteria not set"` until the criteria file is written (section 8) |
| `criteria_version`, `matched_rule` | Which criteria version and which rule (counted from 0) gave the tier; `null` rule means the default tier |
| `model_version` | The saved model that produced the metrics |

The tier is a study-defined prioritization signal for teachers. It is not a diagnosis, and the
teacher decides what to do next.

## 6. Use a saved model without training

Use this when a teammate has already trained the model and you only want to analyze recordings
(for example, to connect the web app).

**No trained Read Smart model exists yet**, because there are no labeled recordings yet. Until
then, the only model you can load is the throwaway one from section 4.

What you need from the teammate who trained it:

| Item | Where it comes from | Where to put it |
|---|---|---|
| The model folder (`analyzer_card.json`, `filler_head.pt`, `clarity_head.joblib`) | `ml/runs/<...>_analyzer_final_test/analyzer/` on their machine | anywhere, e.g. `ml/models/v1.0/` |
| The **same** CNN14 weights file | section 3 | `ml/checkpoints/`, and set `cnn14.checkpoint` |
| The tier criteria file used for that model | `ml/criteria/tier_criteria.yaml` | same place |

Steps:

```bash
# sections 2 and 3 first (set up, CNN14 weights), then:
mkdir -p models/v1.0            # copy the three model files into it
python -m readsmart criteria-check
python -m readsmart analyze --model models/v1.0 --wav path/to/recording.wav
```

Notes:

- No labels, data folder or training step are needed. The timing, filler and clarity settings come
  from the model folder, so results match the trained model.
- The CNN14 file name must match the one the model was trained with, otherwise `analyze` stops.
- Model files are ignored by Git on purpose. A model trained on learners' recordings should be
  shared privately within the team, not committed to this public repository, unless the adviser
  approves.
- From Python (run from `ml/`), this returns the same dictionary:

  ```python
  from readsmart.config import load_config
  from readsmart.analyzer_eval import analyze_file
  result = analyze_file(load_config("config.yaml"), "models/v1.0", "recording.wav")
  ```

## 7. Train the model

### Before you start

Training real models needs things that do not exist yet. List what is still open with:

```bash
python -m readsmart gate
```

That covers the written tier criteria, the labeling rules (including how fillers are marked and
the clarity rating scale), permissions and consent, the agreed target scores, the confirmed stack,
approved storage, and the CNN14 weights. Fill the `gate:` section of `config.yaml` as each is
settled, then set `mode: study`. In study mode every command refuses to run while anything is open.
While `mode: practice`, everything runs but the numbers must not be reported.

### Put the data in place

Git ignores all of this. Never commit it.

```
ml/data/
  raw/                    WAV files, named by anonymized learner code only
  labels.csv              one row per recording        (templates/labels_template.csv)
  filler_marks.csv        start/end of each "uh"/"um"  (templates/filler_marks_template.csv)
  timing_checks.csv       hand-counted pauses/syllables for about 20 recordings
  double_labels.csv       two raters' tiers for a subset
  excluded_clips.csv      optional: clips with the teacher's voice
```

- `labels.csv` needs: `recording_id, learner_code, audio_file, passage_id, session_id, reference_tier, labeler_id, label_version, clarity_score, fillers_marked`.
  Failed first attempts stay in the file with an empty tier.
- **Filler marks are required.** Mark each filler in Audacity (select it, Ctrl+B, type `uh` or `um`),
  export the label track as `<recording_id>.txt` into one folder, then run
  `python -m readsmart import-audacity --labels-dir <that folder>`. Set `fillers_marked` to `yes` for
  every recording that was checked, including ones with no fillers.
- Mark times on the original WAV. The code adjusts for the silence it trims.

### Run the steps

```bash
python -m readsmart qc                  # quality check + cleaning
python -m readsmart split               # locks ~20% of learners as the test set, makes 5 folds
python -m readsmart inventory           # dataset table for Chapter IV
python -m readsmart frames              # CNN14 features (a few minutes on CPU)
python -m readsmart timing-check        # automatic vs hand-timed pauses and syllables
python -m readsmart kappa               # agreement between the two raters
python -m readsmart criteria-check
python -m readsmart analyzer-cv         # cross-validation on the development learners
python -m readsmart analyzer-final --version-label v1.0    # trains the final model, tests ONCE
```

- Tune settings (`timing.*`, `filler.threshold`) using `timing-check` and `analyzer-cv` only. Never
  tune on the final test.
- `split` refuses to redraw the test set, and `analyzer-final` refuses to run twice without
  `--force-reason "..."`, because the test set is meant to be used once.
- The trained model is saved in `runs/<date-time>_study_analyzer_final_test/analyzer/`. That folder
  is what section 6 uses. The same run folder has the scores (`test_scores.json`), the tier table
  and confusion matrix, per-recording predictions, the exact config and the library versions.
- `runs/run_log.csv` keeps one line per run.

Optional comparison baselines (the earlier tier-only classifiers M0 to M5) are described in
[`ml/README.md`](ml/README.md), sections 1 to 9.

## 8. Write the tier criteria

`ml/criteria/tier_criteria.yaml` holds the rules that turn metrics into a tier. It is empty, and
the code never sets a threshold. The researchers write it with the adviser (Objective 4). Format:

```yaml
version: "criteria v1, <date>"
approved_by: "<adviser>"
default_tier: Low            # tier when no rule matches
rules:                       # checked top to bottom, first match wins
  - tier: High
    any:                     # any = at least one condition; all = every condition
      - {metric: clarity_score, op: "<=", value: <number>}
      - {metric: fillers_per_min, op: ">=", value: <number>}
  - tier: Moderate
    any:
      - {metric: pauses_per_min, op: ">=", value: <number>}
```

Check it with `python -m readsmart criteria-check`. Until it has a version and rules, results show
the metrics with `"status": "criteria not set"`.

## 9. Troubleshooting

| Problem | What to do |
|---|---|
| `pip install` cannot find `torch==2.15.0` | Your Python may be too old or too new for that build. Install the CPU build from <https://pytorch.org/get-started/locally/>, then run `pip install -r requirements.txt` again |
| `CNN14 checkpoint not found` | Check `cnn14.checkpoint` in `config.yaml`; the path is relative to that file |
| `Error(s) in loading state_dict` | The weights file is not `Cnn14_16k_mAP=0.438.pth` (for example the 32 kHz version) or the download is incomplete |
| `Readiness gate is not closed` | You are in `mode: study`. Run `python -m readsmart gate` and fill what it lists, or use `mode: practice` for a pipeline check |
| `Format not recognised` when reading audio | The file is not WAV. Convert it to WAV first |
| Recording fails the quality check | Too short, too quiet or clipped. Re-record. The limits are in `audio.qc` and are still to be set from real classroom recordings |
| `Test set is already locked` / `already used` | Intended. See section 7 |
| `No filler-marked recordings among the training learners` | `fillers_marked` is not `yes` for any training recording, or `filler_marks.csv` is missing |

## Data rules

Recordings are from minors. **Never commit** audio, embeddings, model checkpoints, the learner
key, results that contain learner codes, or secrets. `.gitignore` blocks `data/`, `work/`, `runs/`,
`checkpoints/` and audio/model file types; keep it that way. Use anonymized learner codes only.
