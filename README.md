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

## Quick start

```bash
cd ml
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests
python scripts/make_synthetic_data.py --out synthetic
python -m readsmart practice-analyzer --config synthetic/config.yaml
```

## Data rules

Recordings are from minors. **Never commit** audio, embeddings, model checkpoints, the learner
key, results that contain learner codes, or secrets. `.gitignore` blocks `data/`, `work/`, `runs/`,
`checkpoints/` and audio/model file types; keep it that way. Use anonymized learner codes only.
