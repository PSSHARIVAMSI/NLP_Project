# Reddit Mental Health — Multi-Class Topic Classification

Classifies Reddit mental-health posts into **depression, anxiety, crisis,
neutral, loneliness**, comparing five modeling approaches — from a classical
TF-IDF baseline through fine-tuned DistilBERT to zero-shot BART — so the
trade-offs between them can be argued with numbers, not vibes.

## Project structure

```
MentalHealth/
├── app.py                     # Streamlit demo — all 5 models, live audience mode
├── requirements.txt
├── notebooks/
│   └── mental_health_classification.ipynb   # single source of truth: EDA → train → eval → save
├── data/
│   └── processed/              # train / val / test splits used by every model
├── models/                     # trained artifacts app.py loads (gitignored — see below)
│   ├── baseline/                # TF-IDF vectorizer + SGDClassifier
│   ├── rnn/                     # Elman RNN state dict + vocab
│   ├── lstm/                    # BiLSTM state dict + vocab
│   └── distilbert/               # fine-tuned HF model dir
├── reports/                    # confusion matrices + model comparison chart
└── docs/                       # slide deck for the project demo
```

Raw Reddit scrape data, an old EDA notebook, and a duplicate early draft of
the notebook were moved out of this folder to
`../MentalHealth_unused_archive/` — they aren't needed to run or present the
project. Delete that folder once you've confirmed you don't need it.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the models

Run `notebooks/mental_health_classification.ipynb` top to bottom. It streams
and cleans the raw posts, builds the train/val/test split, trains all four
supervised models, evaluates the zero-shot model, and saves everything
`app.py` needs under `models/` and `reports/`.

> **Known gap as of this notebook's last edit:** the Model 5 (zero-shot
> BART) cell and the model-comparison cell that follows it were added but
> the comparison cell was re-run *before* the zero-shot cell was ever
> executed — so `reports/model_comparison.png` and the printed comparison
> table only show 4 models, not 5. Re-run cells 28 onward (zero-shot →
> comparison → "reading the comparison") before using either in the demo.

## Run the app

```bash
streamlit run app.py
```

Models load lazily and are cached per session. If `models/` is empty
(fresh clone, weights gitignored), the app still runs with the zero-shot
model only — it needs no local artifacts.

## Model comparison (test set, from the notebook's last full run)

| Model | Accuracy | Macro F1 | Train time | Inference / sample |
|---|---|---|---|---|
| DistilBERT | 0.617 | 0.613 | ~20 min | ~43 ms |
| TF-IDF + Logistic Regression | 0.565 | 0.563 | ~2 s | ~0.4 ms |
| BiLSTM | 0.522 | 0.515 | ~4 min | ~19 ms |
| RNN | 0.384 | 0.361 | ~1.6 min | ~10 ms |
| Zero-Shot (BART-large-mnli) | *(not yet measured — see gap above)* | | 0 (no training) | slow, CPU-bound |

**Reading it:** TF-IDF wins on accuracy-per-second and is fully
interpretable — the right default for a fast, cheap, explainable baseline.
DistilBERT wins outright on accuracy but costs ~500x the training time and
~100x the inference latency. The RNN underperforms the BiLSTM because a
single-direction Elman recurrence forgets earlier tokens over a ~120-word
post; the BiLSTM's two-directional context largely fixes that. Zero-shot
trades all of that off against zero labeled data and zero retraining —
useful the moment label categories change and you can't wait for a new
annotated dataset.

## Disclaimer

Trained on public Reddit data for a topic-classification exercise. Not a
diagnostic or clinical tool. See the in-app disclaimer for what to do if a
demo surfaces a real crisis disclosure.
