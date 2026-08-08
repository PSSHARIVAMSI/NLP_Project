# NLP_Project — Mental Health Post Classifier

An end-to-end NLP project that classifies Reddit posts into mental-health categories
(`depression`, `anxiety`, `crisis`, `loneliness`, `neutral`) by comparing five modeling
approaches — from a TF-IDF baseline to a fine-tuned DistilBERT and a zero-shot BART model —
and serving the results through a Streamlit demo app.

> **Disclaimer:** This is an capstone project. The classifier is a **topic
> classifier, not a diagnostic or clinical tool**, and must not be used to make decisions
> about anyone's mental health or safety. If a real disclosure of crisis or self-harm risk
> comes up while demoing this, stop and point the person to the **988 Suicide & Crisis
> Lifeline** (call or text 988 in the US) or local emergency services.

## Repository structure

```
NLP_Project/
├── README.md
├── Usecase                          # Short note on how this project maps to NLP/sentiment analysis
└── Mental_Health_Classifier/        # Main project (its own local git repo — see note below)
    ├── Modelling.ipynb              # End-to-end pipeline: load data → EDA → split → train 5 models → compare
    ├── extract_s3_to_proceed.py     # Standalone script: raw per-subreddit S3 CSVs → cleaned, labeled, deduped combined CSV
    ├── app.py                       # Streamlit demo app (see "Streamlit app" section)
    ├── train.csv / val.csv / test.csv   # Saved 70/15/15 split produced by the notebook
    ├── models/                      # Trained model artifacts (see "Trained models" below)
    │   ├── tfidf_vectorizer.joblib
    │   ├── tfidf_classifier.joblib
    │   ├── rnn_state_dict.pt, rnn_vocab.joblib
    │   ├── lstm_state_dict.pt, lstm_vocab.joblib
    │   └── distilbert/               # HF-format saved model (config.json, model.safetensors, tokenizer files)
    ├── reports/                      # Saved PNGs: EDA plots, confusion matrices, training curves, comparison charts
    └── venv/                         # Local Python virtual environment (not portable, not part of the codebase)
```

## Data

- **Source:** Kaggle Reddit Mental Health Dataset, mirrored in an S3 bucket
  (`s3://reddit-mental-health-dataset/`) as per-subreddit, per-month raw CSVs.
- **Label mapping** (subreddit → label), per [`extract_s3_to_proceed.py`](Mental_Health_Classifier/extract_s3_to_proceed.py):

  | Subreddit | Label |
  |---|---|
  | r/depression | depression |
  | r/Anxiety | anxiety |
  | r/SuicideWatch | crisis |
  | r/mentalhealth | neutral |
  | r/lonely | loneliness |

- **Processing pipeline** (`extract_s3_to_proceed.py`): streams every raw CSV from S3,
  cleans text (strips URLs, markdown links, `r/`/`u/` mentions, non-printable characters),
  labels each row from its subreddit (falling back to a filename-prefix heuristic),
  drops junk/empty posts, deduplicates on text, and writes a single combined CSV back to
  `s3://reddit-mental-health-dataset/processed/all_data_combined.csv`.
- **Sampling for training** (`Modelling.ipynb`, cell 1): streams that ~1.3 GB combined CSV
  from S3 in chunks and reservoir-samples up to **20,000 posts per class** (5 classes →
  **100,000 posts total**), to fix a raw-data class imbalance (per the app's leaderboard
  tab, the raw corpus was roughly 34% depression vs. only 9% loneliness before balancing).
- **Split:** stratified 70% train / 15% val / 15% test (`train.csv`, `val.csv`, `test.csv`),
  same split reused for all five models for a fair comparison.

## Notebook pipeline (`Modelling.ipynb`)

The notebook has 7 cells, run top to bottom:

1. **Cell 0** — sets AWS credentials as environment variables.
2. **Cell 1** — imports, config, and streaming load + balanced per-class sampling from S3.
3. **Cell 2** — data quality check: shape, nulls, duplicates, class balance, word/char
   length stats, sample rows per class.
4. **Cell 3** — EDA plots: class distribution bar chart, word-count-per-class box plot,
   per-class word clouds (saved to `reports/`).
5. **Cell 4** — stratified 70/15/15 train/val/test split; saves `train.csv`/`val.csv`/`test.csv`.
6. **Cell 5** — trains and evaluates all 5 models on the same split (details below); saves
   model artifacts to `models/` and confusion matrices / training curves to `reports/`.
7. **Cell 6** — builds a comparison table + bar charts (accuracy, macro F1, inference
   latency) across all 5 models from the results logged in Cell 5.

## Trained models

| # | Model | Approach | Key settings |
|---|---|---|---|
| 1 | **TF-IDF + SGD (log loss)** | `TfidfVectorizer` (max 50K features, 1–2 grams, English stopwords) → `SGDClassifier` with `class_weight="balanced"` | — |
| 2 | **RNN** | Custom `nn.Embedding` + `nn.RNN` + dropout + linear head, trained from scratch on a 20K-word vocab built from the training set | embed dim 128, hidden dim 128, 5 epochs, batch 64, lr 1e-3 |
| 3 | **BiLSTM** | Same setup as the RNN but bidirectional `nn.LSTM`, concatenating both final hidden states | embed dim 128, hidden dim 128, 5 epochs, batch 64, lr 1e-3 |
| 4 | **DistilBERT** | Fine-tuned `distilbert-base-uncased` via 🤗 Transformers, saved in HF format under `models/distilbert/` | max len 128, 2 epochs, batch 16, lr 2e-5, linear warmup schedule |
| 5 | **Zero-Shot BART** | Off-the-shelf `facebook/bart-large-mnli` zero-shot classification pipeline — no fine-tuning | evaluated on 500 test samples, CPU |

All sequence/BERT models use class-weighted cross-entropy to counter any residual class
imbalance. Every model's val/test run produces a confusion-matrix PNG saved to `reports/`.

### Leaderboard (as reported in `app.py`, sourced from the notebook's training run)

| Rank | Model | Macro F1 | Latency / sample | Notes |
|---|---|---|---|---|
| 1 | DistilBERT | 0.631 | ~80–120 ms | Highest accuracy; benefits from GPU |
| 2 | TF-IDF + SGD | 0.605 | < 1 ms | ~4% below BERT; near-zero inference cost |
| 3 | BiLSTM | 0.572 | ~5–15 ms | Beats plain RNN on longer posts (gated memory) |
| 4 | RNN | 0.541 | ~3–10 ms | Vanishing-gradient limits long-range context |
| 5 | Zero-Shot BART | 0.421 | ~400–800 ms | Zero labeled examples used |

Per-class F1 for the best model (DistilBERT), also from `app.py`: crisis 0.78, anxiety 0.65,
depression 0.62, neutral 0.59, loneliness 0.47 (loneliness overlaps heavily with depression
in vocabulary, making it the hardest class).

*I'm quoting these numbers as they appear hardcoded in `app.py`'s leaderboard tab — I did
not independently re-run the notebook to verify them.*

## Streamlit app (`app.py`)

A three-tab demo app:

1. **🎤 Try It Yourself** — paste any text, pick which models to run, get per-model
   predictions with confidence + latency, a confidence bar chart, an ensemble majority
   vote, and a crisis banner (with the 988 lifeline) if any model predicts `crisis`. Also
   includes an optional QR-code generator (via the `qrcode` package) for a deployed app URL.
2. **📊 Model Leaderboard** — the static comparison table and per-class F1 bars described
   above, plus a "training data provenance" table of raw vs. balanced row counts per
   subreddit.
3. **📡 Live Audience Wall** — polls a **published (public CSV) Google Sheet** URL backing
   a Google Form, classifies the latest ~15 responses with all available models, and shows
   them in a table (no login/API key needed since it just reads the published CSV link).

Run it with:

```bash
streamlit run Mental_Health_Classifier/app.py
```

## Setup

The project's `venv/` currently has these packages installed (used for the notebook):
`boto3`, `joblib`, `matplotlib`, `nltk`, `pandas`, `scikit-learn`, `seaborn`, `torch`,
`transformers`, `wordcloud`. There is **no `requirements.txt`** in the repo yet.

To run the notebook, at minimum:

```bash
pip install boto3 pandas scikit-learn torch transformers wordcloud seaborn matplotlib joblib nltk
```

To run `app.py`, you additionally need `streamlit` (required) and, optionally, `qrcode[pil]`
for the QR-code feature — **neither is currently installed in `venv/`**:

```bash
pip install streamlit qrcode[pil]
```

AWS credentials (for `extract_s3_to_proceed.py`) are expected as:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

## How this relates to NLP & sentiment analysis

See [`Usecase`](Usecase) for the full note. In short: this is a text-classification task
that goes beyond standard positive/negative/neutral sentiment analysis into emotion/mental
health-state detection, using standard NLP preprocessing (cleaning, tokenization) across a
spectrum of approaches from classical (TF-IDF) to transformer-based (DistilBERT, zero-shot
BART).

## Credits

Built by **Siva Mani**
Data: Kaggle Reddit Mental Health Dataset.