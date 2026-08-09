"""
Mental Health Post Classifier — Streamlit App
==============================================
Five-model comparison: TF-IDF · RNN · BiLSTM · DistilBERT · Zero-Shot BART

Tabs:
  1. Try It Yourself   — type a post, get predictions from all models
  2. Model Leaderboard — final F1 / latency comparison from the training run
  3. Live Audience Wall — classifies a live Google Form response feed

Run:
    streamlit run app.py
"""

import re
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Sage Minimal theme ─────────────────────────────────────────────────────
SAGE_CSS = """
<style>
/* ── Palette ────────────────────────────────────────────────────────────── */
:root {
    --navy:    #1A2035;
    --sage:    #6B8F71;
    --mint:    #EBF4ED;
    --offwhite:#F7F8FA;
    --text:    #1A2035;
    --muted:   #6B7280;
    --danger:  #D94F3D;
    --warn-bg: #FFF4E5;
    --warn-bd: #F59E0B;
}

/* App background */
.stApp { background-color: var(--offwhite); }

/* ── Theme safety-net ───────────────────────────────────────────────────────
   The app is pinned to a light theme via .streamlit/config.toml, but this
   backs that up: any plain Streamlit-rendered text (labels, markdown blocks,
   checkboxes, inputs) gets an explicit readable color instead of silently
   inheriting whatever the visitor's browser theme happens to be. Elements
   with their own intentional color (hero, footer, crisis banner, cards,
   table) keep their more specific rules further down this stylesheet, which
   win over these lower-specificity, same-order-position rules. */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stWidgetLabel"] p,
[data-testid="stExpander"] summary,
.stCheckbox label p,
.stTextInput label p,
.stTextArea label p,
.stDataFrame,
label {
    color: var(--text);
}

/* Top header bar */
header[data-testid="stHeader"] {
    background-color: var(--navy) !important;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, var(--navy) 0%, #2A3553 100%);
    padding: 2.2rem 2.4rem 1.8rem;
    border-radius: 12px;
    margin-bottom: 1.6rem;
}
.hero h1 {
    color: #FFFFFF;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
    font-family: 'Calibri', 'Segoe UI', sans-serif;
}
.hero p {
    color: #A8B8C8;
    font-size: 0.93rem;
    margin: 0;
    line-height: 1.5;
    font-family: 'Calibri', 'Segoe UI', sans-serif;
}
.hero .pill {
    display: inline-block;
    background: var(--sage);
    color: #fff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: 0.6rem;
}

/* Tab strip */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--navy);
    border-radius: 10px 10px 0 0;
    padding: 0 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    color: #A8B8C8 !important;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.7rem 1.2rem;
    border-bottom: 3px solid transparent;
    background: transparent !important;
    font-family: 'Calibri', 'Segoe UI', sans-serif;
}
.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom: 3px solid var(--sage) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #FFFFFF;
    border-radius: 0 0 10px 10px;
    padding: 1.6rem;
    border: 1px solid #E5E7EB;
    border-top: none;
}

/* Section headings */
.section-head {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--navy);
    margin: 0 0 0.8rem;
    font-family: 'Calibri', 'Segoe UI', sans-serif;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E5E7EB;
    margin-left: 0.5rem;
}

/* Prediction cards */
.pred-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.pred-card {
    background: var(--mint);
    border: 1px solid #C9DFCB;
    border-radius: 10px;
    padding: 1rem 0.9rem;
    text-align: center;
}
.pred-card.crisis {
    background: #FFF0EE;
    border-color: var(--danger);
}
.pred-card .model-name {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
    font-family: 'Calibri', 'Segoe UI', sans-serif;
}
.pred-card .label {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--navy);
    margin-bottom: 0.15rem;
    font-family: 'Calibri', 'Segoe UI', sans-serif;
}
.pred-card .conf {
    font-size: 0.82rem;
    color: var(--sage);
    font-weight: 600;
}
.pred-card .lat {
    font-size: 0.72rem;
    color: var(--muted);
}

/* Crisis banner */
.crisis-banner {
    background: #FFF0EE;
    border-left: 4px solid var(--danger);
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
    font-family: 'Calibri', 'Segoe UI', sans-serif;
    font-size: 0.92rem;
    color: #7B1D1D;
}

/* Leaderboard table */
.lb-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Calibri', 'Segoe UI', sans-serif;
    font-size: 0.88rem;
}
.lb-table th {
    background: var(--navy);
    color: #FFFFFF;
    padding: 0.6rem 0.9rem;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.lb-table td {
    padding: 0.55rem 0.9rem;
    border-bottom: 1px solid #E5E7EB;
    vertical-align: middle;
}
.lb-table tr:last-child td { border-bottom: none; }
.lb-table tr:nth-child(even) td { background: var(--mint); }
.lb-table .winner { color: var(--sage); font-weight: 700; }
.lb-table .badge {
    display: inline-block;
    background: var(--sage);
    color: #fff;
    font-size: 0.68rem;
    padding: 1px 7px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    vertical-align: middle;
    margin-left: 4px;
}
.badge-fast { background: #3B82F6; }
.badge-zs   { background: #7C3AED; }

/* Insight card */
.insight-card {
    background: var(--mint);
    border: 1px solid #C9DFCB;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.7rem;
    font-family: 'Calibri', 'Segoe UI', sans-serif;
}
.insight-card .num {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--sage);
    margin-bottom: 0.15rem;
}
.insight-card p { margin: 0; font-size: 0.9rem; color: var(--navy); line-height: 1.5; }

/* Per-class heatmap row */
.heatmap-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
    font-family: 'Calibri', 'Segoe UI', sans-serif;
    font-size: 0.84rem;
}
.heatmap-label { width: 90px; color: var(--muted); font-weight: 600; }
.heatmap-bar {
    height: 20px;
    border-radius: 4px;
    background: var(--sage);
    transition: width 0.4s ease;
}
.heatmap-val { color: var(--navy); font-weight: 600; }

/* Footer */
.footer {
    margin-top: 2rem;
    padding: 1rem 0;
    border-top: 1px solid #E5E7EB;
    font-size: 0.78rem;
    color: var(--muted);
    font-family: 'Calibri', 'Segoe UI', sans-serif;
    line-height: 1.6;
}

/* Primary button override */
.stButton > button[kind="primary"] {
    background-color: var(--sage) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.8rem !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #547B5A !important;
}
</style>
"""

st.markdown(SAGE_CSS, unsafe_allow_html=True)

# ── Config ─────────────────────────────────────────────────────────────────
# NOTE: all non-DistilBERT artifacts are saved flat under models/ (not in
# baseline/rnn/lstm subfolders) by Modelling.ipynb — paths below match that.
PROJECT_ROOT  = Path(__file__).resolve().parent
MODELS_DIR    = PROJECT_ROOT / "models"
BASELINE_DIR  = MODELS_DIR
RNN_DIR       = MODELS_DIR
LSTM_DIR      = MODELS_DIR
BERT_DIR      = MODELS_DIR / "distilbert"

LABELS   = ["depression", "anxiety", "crisis", "loneliness", "neutral"]
ID2LABEL = dict(enumerate(LABELS))
ZEROSHOT_MODEL_NAME = "facebook/bart-large-mnli"

EMBED_DIM     = 128
HIDDEN_DIM    = 128
MAX_LEN_WORDS = 120
BERT_MAX_LEN  = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── Text cleaning ──────────────────────────────────────────────────────────
URL_RE          = re.compile(r"https?://\S+|www\.\S+")
MD_LINK_RE      = re.compile(r"\[([^\]]*)\]\([^)]*\)")
SUB_MENTION_RE  = re.compile(r"/?r/\w+")
USER_MENTION_RE = re.compile(r"/?u/\w+")
NON_PRINT_RE    = re.compile(r"[^\x20-\x7E\n]")
WHITESPACE_RE   = re.compile(r"\s+")
TOKEN_RE        = re.compile(r"[a-z']+")


def clean_text(text: str) -> str:
    text = str(text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    text = SUB_MENTION_RE.sub(" ", text)
    text = USER_MENTION_RE.sub(" ", text)
    text = NON_PRINT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


# ── Model definitions ──────────────────────────────────────────────────────
# NOTE: attribute names (self.emb / self.drop) must match the names used in
# Modelling.ipynb's RNNClassifier/LSTMClassifier — that's what the saved
# state_dict keys (emb.weight, drop.*, ...) are keyed on. A previous version
# of this file used self.embedding/self.dropout here, which made
# load_state_dict() fail with "Missing key(s) ... Unexpected key(s)" for
# every RNN/BiLSTM prediction.
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn  = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        embedded = self.emb(x)
        packed   = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        return self.fc(self.drop(hidden[-1]))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.emb(x)
        packed   = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        combined = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.drop(combined))


def encode(text: str, vocab: dict, max_len: int = MAX_LEN_WORDS):
    tokens = tokenize(text)[:max_len]
    ids    = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    length = max(len(ids), 1)
    ids    = ids + [vocab["<pad>"]] * (max_len - len(ids))
    return ids, length


# ── Model loaders (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading TF-IDF …")
def load_tfidf():
    vectorizer = joblib.load(BASELINE_DIR / "tfidf_vectorizer.joblib")
    clf        = joblib.load(BASELINE_DIR / "tfidf_classifier.joblib")
    return vectorizer, clf


@st.cache_resource(show_spinner="Loading RNN …")
def load_rnn():
    vocab = joblib.load(RNN_DIR / "rnn_vocab.joblib")
    model = RNNClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, len(LABELS))
    model.load_state_dict(torch.load(RNN_DIR / "rnn_state_dict.pt", map_location=DEVICE))
    return vocab, model.to(DEVICE).eval()


@st.cache_resource(show_spinner="Loading BiLSTM …")
def load_lstm():
    vocab = joblib.load(LSTM_DIR / "lstm_vocab.joblib")
    model = LSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, len(LABELS))
    model.load_state_dict(torch.load(LSTM_DIR / "lstm_state_dict.pt", map_location=DEVICE))
    return vocab, model.to(DEVICE).eval()


@st.cache_resource(show_spinner="Loading DistilBERT …")
def load_bert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(DEVICE).eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading BART zero-shot …")
def load_zeroshot():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=ZEROSHOT_MODEL_NAME, device=-1)


def available_models() -> dict:
    avail = {}
    if (BASELINE_DIR / "tfidf_vectorizer.joblib").exists() and (BASELINE_DIR / "tfidf_classifier.joblib").exists():
        avail["TF-IDF"] = "tfidf"
    if (RNN_DIR / "rnn_state_dict.pt").exists() and (RNN_DIR / "rnn_vocab.joblib").exists():
        avail["RNN"] = "rnn"
    if (LSTM_DIR / "lstm_state_dict.pt").exists() and (LSTM_DIR / "lstm_vocab.joblib").exists():
        avail["BiLSTM"] = "lstm"
    # Require the actual weight file, not just the directory — config.json and
    # tokenizer files can exist (e.g. checked into git) without the ~255MB
    # model.safetensors/pytorch_model.bin being deployed alongside them. This
    # is what previously made DistilBERT look "available" and then crash with
    # OSError as soon as load_bert() actually tried to read the weights.
    if (BERT_DIR / "model.safetensors").exists() or (BERT_DIR / "pytorch_model.bin").exists():
        avail["DistilBERT"] = "bert"
    avail["Zero-Shot (BART)"] = "zeroshot"
    return avail


# ── Prediction engine ──────────────────────────────────────────────────────
def predict_all(text: str, models_to_run: dict) -> pd.DataFrame:
    """Run every requested model and return one row per model.

    Each model runs in its own try/except: if one model's artifact is
    missing, corrupted, or otherwise fails to load/infer, that model is
    skipped (with a warning) instead of taking down the whole page — this
    matters most on the Live Audience Wall, which calls this in a loop over
    many rows with no other error handling around it.
    """
    cleaned = clean_text(text)
    rows    = []

    if "tfidf" in models_to_run.values():
        try:
            vec, clf = load_tfidf()
            t0    = time.time()
            proba = clf.predict_proba(vec.transform([cleaned]))[0]
            lat   = (time.time() - t0) * 1000
            rows.append({"model": "TF-IDF",
                         "prediction": clf.classes_[proba.argmax()],
                         "confidence": float(proba.max()),
                         "latency_ms": lat})
        except Exception as e:
            st.warning(f"TF-IDF prediction failed and was skipped: {e}")

    if "rnn" in models_to_run.values():
        try:
            vocab, model = load_rnn()
            ids, length  = encode(cleaned, vocab)
            t0 = time.time()
            with torch.no_grad():
                logits = model(torch.tensor([ids]).to(DEVICE), torch.tensor([length]))
                proba  = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            lat = (time.time() - t0) * 1000
            rows.append({"model": "RNN",
                         "prediction": ID2LABEL[int(proba.argmax())],
                         "confidence": float(proba.max()),
                         "latency_ms": lat})
        except Exception as e:
            st.warning(f"RNN prediction failed and was skipped: {e}")

    if "lstm" in models_to_run.values():
        try:
            vocab, model = load_lstm()
            ids, length  = encode(cleaned, vocab)
            t0 = time.time()
            with torch.no_grad():
                logits = model(torch.tensor([ids]).to(DEVICE), torch.tensor([length]))
                proba  = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            lat = (time.time() - t0) * 1000
            rows.append({"model": "BiLSTM",
                         "prediction": ID2LABEL[int(proba.argmax())],
                         "confidence": float(proba.max()),
                         "latency_ms": lat})
        except Exception as e:
            st.warning(f"BiLSTM prediction failed and was skipped: {e}")

    if "bert" in models_to_run.values():
        try:
            tokenizer, model = load_bert()
            inputs = tokenizer(cleaned, truncation=True, max_length=BERT_MAX_LEN,
                               return_tensors="pt").to(DEVICE)
            t0 = time.time()
            with torch.no_grad():
                logits = model(**inputs).logits
                proba  = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            lat = (time.time() - t0) * 1000
            id2label = model.config.id2label
            rows.append({"model": "DistilBERT",
                         "prediction": id2label[int(proba.argmax())],
                         "confidence": float(proba.max()),
                         "latency_ms": lat})
        except Exception as e:
            st.warning(f"DistilBERT prediction failed and was skipped: {e}")

    if "zeroshot" in models_to_run.values():
        try:
            zs = load_zeroshot()
            t0 = time.time()
            result = zs(cleaned[:512], candidate_labels=LABELS)
            lat    = (time.time() - t0) * 1000
            rows.append({"model": "Zero-Shot (BART)",
                         "prediction": result["labels"][0],
                         "confidence": float(result["scores"][0]),
                         "latency_ms": lat})
        except Exception as e:
            st.warning(f"Zero-Shot BART prediction failed and was skipped: {e}")

    return pd.DataFrame(rows)


def majority_label(df: pd.DataFrame) -> str:
    return df["prediction"].mode().iloc[0] if not df.empty else ""


# ── Helpers ────────────────────────────────────────────────────────────────
LABEL_EMOJI = {
    "depression": "😔",
    "anxiety":    "😰",
    "crisis":     "🚨",
    "loneliness": "😶",
    "neutral":    "🙂",
}

LABEL_COLOR = {
    "depression": "#6B8F71",
    "anxiety":    "#3B82F6",
    "crisis":     "#D94F3D",
    "loneliness": "#7C3AED",
    "neutral":    "#059669",
}


def render_pred_cards(results: pd.DataFrame):
    """Render one card per model as an HTML grid."""
    cards = ""
    for _, row in results.iterrows():
        is_crisis = row["prediction"] == "crisis"
        emoji     = LABEL_EMOJI.get(row["prediction"], "❓")
        cards += f"""
        <div class="pred-card {'crisis' if is_crisis else ''}">
            <div class="model-name">{row['model']}</div>
            <div class="label">{emoji} {row['prediction'].capitalize()}</div>
            <div class="conf">{row['confidence']:.0%}</div>
            <div class="lat">{row['latency_ms']:.1f} ms</div>
        </div>"""
    st.markdown(f'<div class="pred-grid">{cards}</div>', unsafe_allow_html=True)


def render_crisis_banner():
    st.markdown(
        '<div class="crisis-banner">'
        '🚨 <strong>Crisis flag detected.</strong> '
        'If you or someone you know is in immediate danger, please contact '
        '<strong>988 Suicide &amp; Crisis Lifeline</strong> (call or text 988 in the US) '
        'or your local emergency services.'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="pill">Capstone · NLP · Multi-class Classification</div>
        <h1>🧠 Mental Health Post Classifier</h1>
        <p>
            Five model families — TF-IDF · RNN · BiLSTM · DistilBERT · Zero-Shot BART —
            classifying Reddit posts into <em>depression, anxiety, crisis, loneliness,</em> or <em>neutral</em>.
            Trained on 100 K balanced posts drawn from a 1.55 M-row S3 corpus.
            Topic classifier · not a diagnostic tool.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Model availability check ───────────────────────────────────────────────
avail = available_models()
missing_supervised = [m for m in ["TF-IDF", "RNN", "BiLSTM", "DistilBERT"] if m not in avail]
if missing_supervised:
    st.info(
        f"**Models not yet trained (will be skipped):** {', '.join(missing_supervised)}  \n"
        "Run every cell in `Modelling.ipynb` to generate them."
    )

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎤 Try It Yourself", "📊 Model Leaderboard", "📡 Live Audience Wall"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Try It Yourself
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-head">Type or paste a Reddit post</div>', unsafe_allow_html=True)

    col_input, col_options = st.columns([3, 1])
    with col_input:
        user_text = st.text_area(
            "Post text",
            height=150,
            placeholder="e.g. I've been feeling really isolated lately and can't seem to shake this sadness…",
            label_visibility="collapsed",
            key="single_text",
        )
    with col_options:
        st.markdown("**Select models**")
        model_choices = {}
        for name in avail:
            model_choices[name] = st.checkbox(name, value=True, key=f"chk_{name}")

    selected_models = {k: v for k, v in avail.items() if model_choices.get(k, False)}

    run_btn = st.button("Classify →", type="primary", key="classify_btn")

    if run_btn and user_text.strip():
        if not selected_models:
            st.warning("Select at least one model above.")
        else:
            with st.spinner("Running models …"):
                results = predict_all(user_text, selected_models)

            # Prediction cards
            st.markdown('<div class="section-head">Predictions</div>', unsafe_allow_html=True)
            render_pred_cards(results)

            # Confidence bar chart
            st.markdown('<div class="section-head">Confidence comparison</div>', unsafe_allow_html=True)
            chart_df = results.set_index("model")[["confidence"]].rename(columns={"confidence": "Confidence"})
            st.bar_chart(chart_df, height=200, width="stretch")

            # Latency row
            st.markdown('<div class="section-head">Inference latency</div>', unsafe_allow_html=True)
            lat_df = results[["model", "latency_ms"]].rename(
                columns={"model": "Model", "latency_ms": "Latency (ms)"}
            )
            st.dataframe(
                lat_df.style.format({"Latency (ms)": "{:.1f}"}),
                hide_index=True,
                width="stretch",
            )

            # Ensemble majority vote
            ml = majority_label(results)
            emoji = LABEL_EMOJI.get(ml, "")
            st.success(f"**Ensemble vote:** {emoji} **{ml.capitalize()}**")

            # Crisis banner
            if ml == "crisis" or (results["prediction"] == "crisis").any():
                render_crisis_banner()

    elif run_btn:
        st.warning("Please enter some text before classifying.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Leaderboard (results from Modelling.html training run)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown('<div class="section-head">Overall performance — Macro F1</div>', unsafe_allow_html=True)

    # Results from the training run captured in Modelling.html
    leaderboard_html = """
    <table class="lb-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Model</th>
          <th>Macro F1</th>
          <th>Latency / sample</th>
          <th>Training data</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td class="winner">DistilBERT <span class="badge">Best</span></td>
          <td class="winner">0.631</td>
          <td>~80–120 ms</td>
          <td>70 K rows (supervised)</td>
          <td>Highest accuracy; needs GPU for fast inference</td>
        </tr>
        <tr>
          <td>2</td>
          <td>TF-IDF + SGD <span class="badge badge-fast">Fastest</span></td>
          <td>0.605</td>
          <td>&lt; 1 ms</td>
          <td>70 K rows (supervised)</td>
          <td>~4% below BERT; near-zero inference cost</td>
        </tr>
        <tr>
          <td>3</td>
          <td>BiLSTM</td>
          <td>0.572</td>
          <td>~5–15 ms</td>
          <td>70 K rows (supervised)</td>
          <td>Better than RNN on long posts; gated memory</td>
        </tr>
        <tr>
          <td>4</td>
          <td>RNN</td>
          <td>0.541</td>
          <td>~3–10 ms</td>
          <td>70 K rows (supervised)</td>
          <td>Vanishing-gradient limits long-range context</td>
        </tr>
        <tr>
          <td>5</td>
          <td>Zero-Shot BART <span class="badge badge-zs">No labels</span></td>
          <td>0.421</td>
          <td>~400–800 ms</td>
          <td>0 labeled examples</td>
          <td>F1 gap vs. BERT = cost of having no training data</td>
        </tr>
      </tbody>
    </table>
    """
    st.markdown(leaderboard_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">Per-class F1 — Best model (DistilBERT)</div>', unsafe_allow_html=True)

    # Per-class F1 from notebook insights
    per_class = {
        "crisis":     0.78,
        "anxiety":    0.65,
        "depression": 0.62,
        "neutral":    0.59,
        "loneliness": 0.47,
    }
    max_f1 = max(per_class.values())

    bars_html = ""
    for label, f1 in per_class.items():
        width_pct = int((f1 / max_f1) * 100 * 0.85)
        emoji = LABEL_EMOJI.get(label, "")
        bars_html += f"""
        <div class="heatmap-row">
            <span class="heatmap-label">{emoji} {label.capitalize()}</span>
            <div class="heatmap-bar" style="width:{width_pct}%; background:{LABEL_COLOR.get(label, '#6B8F71')};"></div>
            <span class="heatmap-val">{f1:.2f}</span>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">Key insights</div>', unsafe_allow_html=True)

    insights = [
        ("01", "DistilBERT is the strongest overall (Macro F1 0.631) — worth the inference cost when accuracy is the priority."),
        ("02", "TF-IDF is the pragmatic production choice: only ~4% below DistilBERT, trains in under 10 seconds, runs at sub-millisecond latency."),
        ("03", "BiLSTM consistently beats the plain RNN on longer posts — gated memory cells handle long-range dependencies that vanilla RNNs can't."),
        ("04", "Zero-Shot BART reaches F1 0.421 with zero labeled examples. The gap to DistilBERT (0.631) is the exact cost of having no training data — essential for new label sets or cold-start domains."),
        ("05", "Crisis is the easiest class to classify (F1 0.78) — distinctive self-harm vocabulary makes it stand out. Loneliness is the hardest (F1 0.47) — its language overlaps heavily with depression."),
        ("06", "Class balancing was critical. The raw 1.55 M-row corpus had depression at 34 % and loneliness at only 9 %. Balanced 20 K/class sampling (100 K total) eliminated that skew before training."),
    ]
    for num, text in insights:
        st.markdown(
            f'<div class="insight-card"><div class="num">{num}</div><p>{text}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">Training data provenance</div>', unsafe_allow_html=True)

    provenance_data = {
        "Subreddit":        ["r/depression", "r/Anxiety", "r/SuicideWatch", "r/mentalhealth", "r/lonely"],
        "Label":            ["depression", "anxiety", "crisis", "neutral", "loneliness"],
        "Raw rows":         ["~470 K", "~280 K", "~310 K", "~340 K", "~150 K"],
        "Balanced sample":  ["20 K", "20 K", "20 K", "20 K", "20 K"],
    }
    st.dataframe(pd.DataFrame(provenance_data), hide_index=True, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Audience Wall
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-head">Live feed from a Google Form</div>', unsafe_allow_html=True)

    with st.expander("Setup instructions (one-time, ~2 minutes)", expanded=False):
        st.markdown(
            """
1. Create a **Google Form** with a single short-answer question  
   *(e.g. "Write a sentence about how you're feeling")*.
2. Open the linked **Responses** Google Sheet.
3. **File → Share → Publish to web** → publish as **CSV**.
4. Paste that CSV link below — no login or API key required.
5. Share the **Form** link (or a QR code) with your audience.  
   This panel polls the sheet and classifies each new row with all five models.
            """
        )

    sheet_url     = st.text_input("Published Google Sheet CSV URL", placeholder="https://docs.google.com/.../pub?output=csv", key="sheet_url")
    col_a, col_b  = st.columns([1, 1])
    refresh       = col_a.button("🔄 Refresh now", key="refresh_wall")
    text_col_name = col_b.text_input("Response column name", value="Write a sentence about how you're feeling")

    if sheet_url.strip():
        try:
            responses = pd.read_csv(sheet_url.strip())
        except Exception as e:
            st.error(f"Couldn't read sheet as CSV: {e}")
            responses = None

        if responses is not None:
            if text_col_name not in responses.columns:
                st.warning(
                    f"Column **'{text_col_name}'** not found.  \n"
                    f"Available columns: {list(responses.columns)}"
                )
            else:
                latest = responses.dropna(subset=[text_col_name]).tail(15)
                if latest.empty:
                    st.info("No responses yet — waiting for the audience to submit.")
                else:
                    with st.spinner("Classifying latest responses …"):
                        rows = []
                        for _, row in latest.iloc[::-1].iterrows():
                            post_text = str(row[text_col_name])
                            preds     = predict_all(post_text, avail)
                            row_out   = {"text": post_text[:120] + ("…" if len(post_text) > 120 else "")}
                            for _, r in preds.iterrows():
                                row_out[r["model"]] = f"{LABEL_EMOJI.get(r['prediction'], '')} {r['prediction']} ({r['confidence']:.0%})"
                            rows.append(row_out)
                        wall_df = pd.DataFrame(rows)

                    st.dataframe(wall_df, hide_index=True, width="stretch")

                    model_cols = [c for c in wall_df.columns if c != "text"]
                    if wall_df[model_cols].apply(lambda col: col.str.contains("crisis")).any().any():
                        render_crisis_banner()
    else:
        st.info("Paste a published Google Sheet CSV link above to start the live wall.")


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        <strong>Disclaimer:</strong> This classifier was trained on public Reddit data for an academic capstone project.
        It is a <em>topic classifier</em>, not a diagnostic or clinical tool, and must not be used to make decisions
        about anyone's mental health or safety. If this demo surfaces a real disclosure of crisis or self-harm risk
        from an audience member, pause the demo and direct them to a crisis line —
        <strong>988 Suicide &amp; Crisis Lifeline</strong> (call or text 988, US) · or local emergency services.<br><br>
        Built by <strong>Siva Mani</strong> ·
        Models: TF-IDF · RNN · BiLSTM · DistilBERT · Zero-Shot BART-large-mnli ·
        Data: Kaggle Reddit Mental Health Dataset
    </div>
    """,
    unsafe_allow_html=True,
)