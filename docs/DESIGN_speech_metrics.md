# Design note: one speech-metrics model + separate tier step

Proposed November 15, 2026. Theo asked for this design and chose filler detection as **required**.
Nothing here has been reviewed by Sir Heubert yet. Wording below is a draft for the team.

## 1. What changed

| Before (Training Plan, Nov 15) | Now |
|---|---|
| CNN14 features go straight into a classifier that outputs Low / Moderate / High | One **speech analyzer** outputs speech metrics for each recording. A separate **tier step** turns those metrics into Low / Moderate / High using written rules |
| Output: tier only | Output: metrics + tier + criteria version |
| Tier criteria were needed only to label recordings | Tier criteria now *are* the rules the system runs, written as thresholds on named metrics |

The earlier direct-tier models (M0 to M5) stay in the code as comparison baselines. Whether to
report them is open (see section 7).

## 2. The speech analyzer (one model)

All parts share the frozen PANNs CNN14 front end (64-band log-mel, 16 kHz).

| Output | How it is produced | Labels needed for training |
|---|---|---|
| duration_s, speaking_time_s, speaking_ratio | Loudness-based silence detection | None |
| pause_count, pauses_per_min, total_pause_s, mean_pause_s, longest_pause_s | Silences of at least `min_pause_s` between speech | None (checked against hand timing) |
| syllable_count, speech_rate_syl_per_s, articulation_rate_syl_per_s | Loudness peaks in voiced speech (de Jong & Wempe, 2009 [VERIFY]); peaks inside detected fillers are not counted | None (checked against hand counts) |
| filler_count, fillers_per_min, filler_time_s | **Filler head**: small 1-D CNN on CNN14 frame features (80 ms frames), then frames joined into events | Start and end time of every "uh"/"um" in a subset of recordings (**required**) |
| clarity_score | **Clarity head**: ridge regression on pooled CNN14 frame features | A rater's clarity score per recording on a written scale |

None of these are words correct per minute or word accuracy. The system does not know which words
were read.

## 3. The tier step

- Rules live in `criteria/tier_criteria.yaml`, written by the researchers with the adviser
  (Objective 4). The file is empty now and the code never fills in a number.
- Rules are checked top to bottom and the first match wins, for example "High if clarity_score is
  at most X, or fillers_per_min is at least Y". A default tier applies when no rule matches.
- Every result stores the criteria version, so old results can be traced if the criteria change.
- With no criteria, the system still returns the metrics, with the tier shown as "criteria not set".

## 4. Labels and data the team must produce

| Item | Status | Why |
|---|---|---|
| Filler marking protocol (what counts as a filler, tool, who marks, how disagreements are settled) | **[TO BE FILLED], required** | The filler head can only learn from these marks |
| Filler marks for a subset of recordings (Audacity label track; `import-audacity` converts them) | **[TO BE FILLED], required** | Training and testing the filler head. Recordings that were checked and have no fillers still count (`fillers_marked = yes`) |
| Clarity rating scale and rater instructions | [TO BE FILLED] | Training the clarity head; sets `clarity.scale_min/max` |
| Clarity score for each recording | [TO BE FILLED] | Same |
| Hand-timed pauses and syllable counts on about 20 recordings | [TO BE FILLED] | Checking and tuning the timing settings (`timing-check`) |
| Reference tier per recording | Unchanged | Evaluating the tier step end to end |
| Tier criteria as metric thresholds | [TO BE FILLED] G1 | Running the tier step |

[DECIDE] how many recordings need filler marks. It must be enough that the locked test set still
contains marked fillers after the split.

## 5. Evaluation (technical only)

| What | Metric | Command |
|---|---|---|
| Pauses, syllable peaks | Mean absolute error, bias, Pearson r and Spearman rho against hand timing (development learners only) | `timing-check` |
| Filler detection | Event-level precision, recall and F1 (a detection matches a marked filler when they overlap by at least half of the shorter one), frame-level F1, and error in filler count | `analyzer-cv`, `analyzer-final` |
| Clarity score | Mean absolute error, bias, Pearson r and Spearman rho against rater scores | same |
| Tier | Accuracy, precision, recall and F1 per tier, macro and weighted averages, confusion matrix | same, once criteria exist |

Splits are unchanged: learner-level locked test set and grouped 5-fold CV. The test set is used once.

## 6. What it changes in the thesis (drafts for the team)

**Objectives (Chapter I).** Proposed wording:
- 2. Develop a CNN-based speech analyzer that computes speech metrics from the spectrogram:
  pauses, speech and articulation rate, filler occurrences, and a clarity score.
- 3. Develop a tier assignment component that classifies each recording as Low, Moderate or High
  risk from the speech metrics, using the documented criteria.
- 4. Establish and document the criteria, expressed as thresholds on the speech metrics.
- 6. Evaluate technical performance: accuracy, precision, recall and F1-score for filler detection
  and for the tier assignment, and agreement with hand measurements and rater scores for the
  timing measures and clarity score.

**Scope, framing and Definition of Terms.** The project rules say the system "does not report
specific error types" and "is not a fluency scorer". Pauses, speech rate and fillers are
fluency-type measures, so those lines need new wording. A possible version: *"The system reports
general speech measures (pauses, speech rate, filler occurrences and a clarity score) to support
the tier. It does not measure words correct per minute or word accuracy, does not identify
mispronounced words, and does not diagnose any speech, language or learning disorder."*
New terms to define: Speech Metrics, Pause, Filler, Speech Rate, Articulation Rate, Clarity Score,
Tier Criteria. **Needs Sir Heubert's approval.**

**Chapter III.**
- Program workflow step 7: "the system analyzes the recording" becomes two steps: speech analysis,
  then tier assignment.
- Module list (Table 3.30): split the Classification Service into **Speech Analysis** (metrics) and
  **Tier Assignment** (rules). Add **Criteria Management** (Administrator uploads a new criteria
  version and sets it active) and filler/clarity labeling in Model Management.
- UC7 Classify Recording: main flow returns metrics and tier. Add a use case for managing criteria versions.
- Storyboard Screen 5 (result): show the tier, the metric values, and the criteria version, with
  the non-diagnostic note.
- Figure 3.19 and the training plan's model table: update to the analyzer + tier step.

**ERD and Data Dictionary.**

| New / changed entity | Fields |
|---|---|
| SPEECH_METRICS (new, 1:1 with CLASSIFICATION_RESULT) | metrics_id PK, result_id FK, duration_s, speaking_time_s, pause_count, total_pause_s, mean_pause_s, longest_pause_s, syllable_count, speech_rate, articulation_rate, filler_count, filler_time_s, clarity_score |
| TIER_CRITERIA (new) | criteria_id PK, version_label, rules (JSON text), approved_by, approved_at, is_active |
| CLASSIFICATION_RESULT (changed) | add criteria_id FK; tier is nullable while criteria are not set |
| FILLER_MARK (new, training only) | mark_id PK, recording_id FK, start_s, end_s, label, marked_by |
| RECORDING (changed, training only) | add clarity_score (nullable), clarity_rated_by, fillers_marked (boolean) |
| MODEL_VERSION (changed) | classifier_type becomes "Speech analyzer"; add filler_f1 and clarity_mae |

Relationships: TIER_CRITERIA 1 to many CLASSIFICATION_RESULT; RECORDING 1 to many FILLER_MARK.

**Training plan.** Section 3.3 gains filler marking and clarity rating. Table 4 changes (analyzer
plus baselines). Section 8 gains the metrics in section 5 above. The readiness gate gains the
filler marking protocol, the clarity scale and the criteria file.

## 7. Open decisions

- [DECIDE] Number of recordings with filler marks and clarity ratings, and who rates them.
- [DECIDE] Filler threshold (`filler.threshold`), chosen on CV only. Also the timing settings
  (`timing.*`), tuned with `timing-check` on development learners only.
- [DECIDE] Whether the earlier direct-tier classifiers (M0 to M5) are reported as baselines.
- [DECIDE] Whether pause and rate measures should also have their own reported targets on the Validation Board.
