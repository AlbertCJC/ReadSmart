# Synthetic pipeline test, November 15, 2026

> **These numbers are NOT results.** The "recordings" are synthetic buzzy tone sequences, not
> speech and not children. Tier labels were assigned by the generator (High files simply have
> more pauses and noise). CNN14 ran with random weights because no checkpoint was available.
> The only purpose was to check that every step of the code runs and that the safeguards work.
> Never cite, report or plot any of this in the thesis.

Environment: Python 3.13, torch 2.15.0 (CPU, 4 cores), librosa 1.0.1, scikit-learn 1.9.1, torchlibrosa 0.1.0.

## What was checked

| Check | Outcome |
|---|---|
| 38 synthetic files for 36 fake learners (17 Low, 12 Moderate, 7 High) | Generated |
| Quality check | 36 passed; the 2 deliberately bad files failed ("too short", "too quiet") and were left out |
| Clips (3 / 5 / 10 s) | 314 / 197 / 110 kept; 69 / 31 / 8 mostly-silent clips dropped and logged |
| CNN14 embeddings with 2 augmented copies per clip | Cached for all three clip lengths (about 2.5 min on CPU) |
| Locked test set | 7 learners (3 Low, 2 Moderate, 2 High); a second `split` refuses to redraw it |
| Grouped 5-fold CV on the other 29 learners | Every fold has every tier; leakage check passed on every fit |
| Cross-validation | 101 configurations of M0 to M4 (about 5 min) plus M5 with one reduced setting on 10 s clips (about 3.5 min) |
| Selection rule | Chose M2 over M3 because it was within one SD and simpler; noted that the High-recall floor is not set yet |
| Final test | Ran once; a second run was refused until a reason was given, and the reason went into the run log |
| Outputs | Table 5 CSVs, confusion matrix PNGs, predictions, model file, MODEL_VERSION model card |
| `predict` on single files | Returned a tier for a good file and "failed: too short" for a bad one |
| `figure` | Saved a 64-band log-mel image |
| Shuffled labels (leakage sanity check) | M2 macro F1 fell to 0.17 to 0.21 on CV, at or below chance, as expected when labels carry no information |
| Study mode with the gate open | Refused to run and listed the 9 open items |
| Unit tests | 6 passed |

## Timing guidance for the real run

On a 4-core laptop CPU, embeddings and M0 to M4 take minutes. M5 (the from-scratch CNN) took
about 12 minutes per setting on 3 s clips with 60 epochs, so the full M5 grid (3 clip lengths x
4 settings x 5 folds) would take hours on CPU. Run M5 on a GPU, or narrow its grid with
`models.M5.clip_lengths_s` once the clip length for M2 is chosen. Whether learner audio may go
to a cloud GPU is gate item G10.

## Not verified here

- Loading the real PANNs checkpoint: the download host was not reachable from this environment.
  The CNN14 layers were copied from the PANNs source with the same parameter names, and loading
  uses `strict=True`, so a mismatch will stop with an error on the first `embed` run.
- Real recordings: thresholds for the quality check and silence trimming must be set from a few
  real classroom recordings.

## Speech analyzer test (same day, after the redesign)

Synthetic files now include flat "filler" tones with known start and end times, fake 1 to 5 clarity
scores, and the generator's own pause and syllable counts. Fillers were "marked" in about two thirds
of the recordings. The tier step used `criteria_synthetic.yaml`, made-up rules for this test only.
Same caveats as above: **not results**.

| Check | Outcome |
|---|---|
| `practice-analyzer` (QC, split, frame features, CV, final test, saved model) | Ran in about 1 min 40 s on CPU |
| Filler marks shifted for trimmed leading silence | Detected events lined up with the marks |
| Analyzer CV and final test | Filler event P/R/F1, frame F1, count error, clarity MAE and correlation, and tier metrics all computed; test set lock works |
| `analyze` on one file | Returned the 15 metrics, the tier, the matched rule and the criteria version |
| Empty real criteria file | `criteria-check` says "criteria not set"; results then carry no tier |
| `timing-check` | Pause counts matched the generator exactly. Syllable counts were too low on the noisiest files because the voicing filter rejects noise-heavy fake syllables. This is the kind of problem `timing-check` exists to catch: on real recordings, tune `timing.*` (including `voiced_only`) on development learners before training |
| Unit tests | 10 passed |
