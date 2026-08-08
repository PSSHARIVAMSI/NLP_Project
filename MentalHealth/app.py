"""Mental Health Post Classifier — Streamlit demo.

Loads the models trained in notebooks/mental_health_classification.ipynb
(TF-IDF, RNN, LSTM, DistilBERT, plus zero-shot BART-large-mnli) and offers
two ways to feed it text:

  Tab 1 "Try It Yourself"     — anyone with the app's URL types their own
                                 post and immediately sees all four models'
                                 predictions side by side, on their own
                                 screen. Simplest live-audience setup: put
                                 the deployed app's URL behind a QR code
                                 (this tab generates one for you) and let
                                 people open it on their phones.

  Tab 2 "Live Audience Wall"  — for showing everyone's submissions on the
                                 presenter's screen at once. Point it at a
                                 Google Form's response sheet (published to
                                 the web as CSV, no login/API key needed)
                                 and it polls for new rows and classifies
                                 each one with all four models.

Run with:
    streamlit run app.py
"""

import io
import re
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Config — must match mental_health_sentiment_analysis.ipynb
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_DIR = MODELS_DIR / "baseline"
RNN_DIR = MODELS_DIR / "rnn"
LSTM_DIR = MODELS_DIR / "lstm"
BERT_DIR = MODELS_DIR / "distilbert"

LABELS = ["depression", "anxiety", "crisis", "neutral", "loneliness"]
ID2LABEL = dict(enumerate(LABELS))
ZEROSHOT_MODEL_NAME = "facebook/bart-large-mnli"

EMBED_DIM = 128
HIDDEN_DIM = 128
MAX_LEN_WORDS = 120
BERT_MAX_LEN = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# --------------------------------------------------------------------------
# Text cleaning / tokenizing — identical logic to the notebook
# --------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
SUB_MENTION_RE = re.compile(r"/?r/\w+")
USER_MENTION_RE = re.compile(r"/?u/\w+")
NON_PRINTABLE_RE = re.compile(r"[^\x20-\x7E\n]")
WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z']+")


def clean_text(text: str) -> str:
    text = str(text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    text = SUB_MENTION_RE.sub(" ", text)
    text = USER_MENTION_RE.sub(" ", text)
    text = NON_PRINTABLE_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------
# RNN / LSTM model definitions — identical architecture to the notebook
# --------------------------------------------------------------------------
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        return self.fc(self.dropout(hidden[-1]))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        combined = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(combined))


def encode(text: str, vocab: dict, max_len: int = MAX_LEN_WORDS):
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    length = max(len(ids), 1)
    ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    return ids, length


# --------------------------------------------------------------------------
# Model loading (cached so Streamlit only loads each model once per session)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading TF-IDF model...")
def load_tfidf():
    vectorizer = joblib.load(BASELINE_DIR / "tfidf_vectorizer.joblib")
    clf = joblib.load(BASELINE_DIR / "sgd_classifier.joblib")
    return vectorizer, clf


@st.cache_resource(show_spinner="Loading RNN model...")
def load_rnn():
    vocab = joblib.load(RNN_DIR / "vocab.joblib")
    model = RNNClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, len(LABELS))
    model.load_state_dict(torch.load(RNN_DIR / "rnn_state_dict.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return vocab, model


@st.cache_resource(show_spinner="Loading LSTM model...")
def load_lstm():
    vocab = joblib.load(LSTM_DIR / "vocab.joblib")
    model = LSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, len(LABELS))
    model.load_state_dict(torch.load(LSTM_DIR / "lstm_state_dict.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return vocab, model


@st.cache_resource(show_spinner="Loading DistilBERT model...")
def load_bert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(DEVICE).eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading zero-shot classifier (BART-large-mnli)...")
def load_zeroshot():
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=ZEROSHOT_MODEL_NAME, device=-1)


def available_models():
    avail = {}
    if (BASELINE_DIR / "tfidf_vectorizer.joblib").exists():
        avail["TF-IDF"] = "tfidf"
    if (RNN_DIR / "rnn_state_dict.pt").exists():
        avail["RNN"] = "rnn"
    if (LSTM_DIR / "lstm_state_dict.pt").exists():
        avail["LSTM"] = "lstm"
    if BERT_DIR.exists() and any(BERT_DIR.iterdir()):
        avail["DistilBERT"] = "bert"
    # Zero-shot needs no local artifacts — it's a pretrained NLI model pulled
    # from the HF hub on first use, so it's always offered as an option.
    avail["Zero-Shot (BART)"] = "zeroshot"
    return avail


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_all(text: str, models_to_run: dict) -> pd.DataFrame:
    cleaned = clean_text(text)
    rows = []

    if "tfidf" in models_to_run.values():
        vectorizer, clf = load_tfidf()
        t0 = time.time()
        proba = clf.predict_proba(vectorizer.transform([cleaned]))[0]
        latency = (time.time() - t0) * 1000
        pred = clf.classes_[proba.argmax()]
        rows.append({"model": "TF-IDF", "prediction": pred, "confidence": proba.max(), "latency_ms": latency})

    if "rnn" in models_to_run.values():
        vocab, model = load_rnn()
        ids, length = encode(cleaned, vocab)
        t0 = time.time()
        with torch.no_grad():
            logits = model(torch.tensor([ids]).to(DEVICE), torch.tensor([length]))
            proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        latency = (time.time() - t0) * 1000
        rows.append({"model": "RNN", "prediction": ID2LABEL[int(proba.argmax())], "confidence": proba.max(), "latency_ms": latency})

    if "lstm" in models_to_run.values():
        vocab, model = load_lstm()
        ids, length = encode(cleaned, vocab)
        t0 = time.time()
        with torch.no_grad():
            logits = model(torch.tensor([ids]).to(DEVICE), torch.tensor([length]))
            proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        latency = (time.time() - t0) * 1000
        rows.append({"model": "LSTM", "prediction": ID2LABEL[int(proba.argmax())], "confidence": proba.max(), "latency_ms": latency})

    if "bert" in models_to_run.values():
        tokenizer, model = load_bert()
        inputs = tokenizer(cleaned, truncation=True, max_length=BERT_MAX_LEN, return_tensors="pt").to(DEVICE)
        t0 = time.time()
        with torch.no_grad():
            logits = model(**inputs).logits
            proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        latency = (time.time() - t0) * 1000
        id2label = model.config.id2label
        rows.append({"model": "DistilBERT", "prediction": id2label[int(proba.argmax())], "confidence": proba.max(), "latency_ms": latency})

    if "zeroshot" in models_to_run.values():
        zs_pipeline = load_zeroshot()
        t0 = time.time()
        result = zs_pipeline(cleaned[:512], candidate_labels=LABELS)
        latency = (time.time() - t0) * 1000
        rows.append({
            "model": "Zero-Shot (BART)",
            "prediction": result["labels"][0],
            "confidence": result["scores"][0],
            "latency_ms": latency,
        })

    return pd.DataFrame(rows)


def majority_label(results_df: pd.DataFrame) -> str:
    if results_df.empty:
        return ""
    return results_df["prediction"].mode().iloc[0]


def crisis_note():
    st.warning(
        "One or more models flagged this as **crisis**-related. If you or "
        "someone you know is in immediate danger, please contact local "
        "emergency services or a crisis line (e.g. 988 Suicide & Crisis "
        "Lifeline in the US)."
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Mental Health Post Classifier", page_icon="🧠", layout="wide")
st.title("🧠 Mental Health Post Classifier")
st.caption(
    "Compares TF-IDF, RNN, LSTM, DistilBERT, and a zero-shot BART model on "
    "the same text — trained on r/depression, r/Anxiety, r/SuicideWatch, "
    "r/mentalhealth, r/lonely posts. Topic classifier, not a diagnostic "
    "tool — see disclaimer below."
)

avail = available_models()
supervised_avail = {k: v for k, v in avail.items() if v != "zeroshot"}
if not supervised_avail:
    st.error(
        "No trained models found under `models/`. Run every cell of "
        "`notebooks/mental_health_classification.ipynb` first — it saves "
        "the four supervised models this app loads (zero-shot needs no "
        "local artifacts and stays available on its own)."
    )

missing = [m for m in ["TF-IDF", "RNN", "LSTM", "DistilBERT"] if m not in avail]
if missing:
    st.info(f"Not yet trained (skipping): {', '.join(missing)}")

tab1, tab2 = st.tabs(["🎤 Try It Yourself", "📡 Live Audience Wall"])

# ---- Tab 1: single text box, all models -----------------------------------
with tab1:
    st.subheader("Type a post, compare all four models")
    text = st.text_area("Post text", height=140, placeholder="Type or paste a post here...", key="single_text")

    if st.button("Classify", type="primary", key="classify_btn") and text.strip():
        with st.spinner("Running models..."):
            results = predict_all(text, avail)

        st.dataframe(
            results.style.format({"confidence": "{:.1%}", "latency_ms": "{:.1f} ms"}),
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(results.set_index("model")["confidence"])

        if majority_label(results) == "crisis" or (results["prediction"] == "crisis").any():
            crisis_note()

    st.divider()
    st.subheader("📱 Put this in front of an audience")
    st.markdown(
        "Deploy this app (e.g. `streamlit run app.py` on a machine reachable "
        "over the network, or push it to **Streamlit Community Cloud** for a "
        "public URL) and share that URL as a QR code — each attendee opens "
        "it on their own phone, types their own post, and immediately sees "
        "their own multi-model results. No Google Forms or extra services "
        "needed for this mode."
    )
    deployed_url = st.text_input("Paste your deployed app's URL to generate a QR code", placeholder="https://your-app.streamlit.app")
    if deployed_url.strip():
        try:
            import qrcode

            qr_img = qrcode.make(deployed_url.strip())
            buf = io.BytesIO()
            qr_img.save(buf, format="PNG")
            st.image(buf.getvalue(), caption=deployed_url, width=220)
        except ImportError:
            st.info("Install `qrcode` (`pip install qrcode[pil]`) to render a QR code here, or use any free online QR generator with this URL.")

# ---- Tab 2: live audience wall ---------------------------------------------
with tab2:
    st.subheader("Live feed from a Google Form")
    st.markdown(
        """
    **Setup (one-time, ~2 minutes):**
    1. Create a Google Form with a single short-answer question (e.g. *"Write a sentence about how you're feeling"*).
    2. Open the linked **Responses** Google Sheet.
    3. `File → Share → Publish to web` → publish the sheet as **CSV**.
    4. Paste that published CSV link below. No login or API key required — it's a public read-only CSV export.
    5. Share the **Form** link (or a QR code to it) with your audience; this panel polls the sheet and classifies each new response with all four models.
        """
    )

    sheet_url = st.text_input("Published Google Sheet CSV URL", placeholder="https://docs.google.com/.../pub?output=csv", key="sheet_url")
    col_a, col_b = st.columns([1, 1])
    refresh = col_a.button("🔄 Refresh now")
    text_col_name = col_b.text_input("Response column name in the sheet", value="Write a sentence about how you're feeling")

    if sheet_url.strip():
        try:
            responses = pd.read_csv(sheet_url.strip())
        except Exception as e:
            st.error(f"Couldn't read that sheet as CSV: {e}")
            responses = None

        if responses is not None:
            if text_col_name not in responses.columns:
                st.warning(f"Column '{text_col_name}' not found. Available columns: {list(responses.columns)}")
            else:
                latest = responses.dropna(subset=[text_col_name]).tail(15)
                if latest.empty:
                    st.info("No responses yet — waiting for the audience to submit.")
                else:
                    with st.spinner("Classifying latest responses..."):
                        rows = []
                        for _, row in latest.iloc[::-1].iterrows():
                            submitted_text = str(row[text_col_name])
                            preds = predict_all(submitted_text, avail)
                            row_out = {"text": submitted_text}
                            for _, r in preds.iterrows():
                                row_out[r["model"]] = f"{r['prediction']} ({r['confidence']:.0%})"
                            rows.append(row_out)
                        wall_df = pd.DataFrame(rows)
                    st.dataframe(wall_df, hide_index=True, use_container_width=True)

                    model_cols = [c for c in wall_df.columns if c != "text"]
                    if wall_df[model_cols].apply(lambda col: col.str.startswith("crisis")).any().any():
                        crisis_note()
    else:
        st.info("Paste a published Google Sheet CSV link above to start the live wall.")

st.divider()
st.caption(
    "Disclaimer: trained on public Reddit data for a topic-classification "
    "exercise. It is not a diagnostic or clinical tool and should not be "
    "used to make decisions about anyone's mental health or safety. If "
    "this demo surfaces a real disclosure of crisis/self-harm risk from an "
    "audience member, pause the demo and follow up with them directly and "
    "point them to a crisis line."
)
