"""
extract_s3_to_processed.py

Streams raw Reddit mental-health CSVs directly from S3:
    s3://reddit-mental-health-dataset/{year}/{month}/*.csv

Cleans, labels, deduplicates, and writes the result back to:
    s3://reddit-mental-health-dataset/processed/all_data_combined.csv

Prerequisites:
    pip install boto3 pandas

Credentials (set these in your terminal before running):
    export AWS_ACCESS_KEY_ID=your_access_key
    export AWS_SECRET_ACCESS_KEY=your_secret_key
    export AWS_DEFAULT_REGION=us-east-1

Compatible with Python 3.9+
"""

import io
import re
import os
import logging
from collections import Counter
from typing import Optional, List   # Python 3.9 compatible

import boto3
import pandas as pd
from botocore.exceptions import ClientError

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BUCKET      = "reddit-mental-health-dataset"
RAW_PREFIX  = ""                               # keys like: 2019/Aug/depjaug19.csv
OUTPUT_KEY  = "processed/all_data_combined.csv"
AWS_REGION  = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

SUBREDDIT_TO_LABEL = {
    "depression":   "depression",
    "anxiety":      "anxiety",
    "suicidewatch": "crisis",
    "mentalhealth": "neutral",
    "lonely":       "loneliness",
}

FILENAME_PREFIX_TO_LABEL = {
    "dep": "depression",
    "ani": "anxiety",
    "sw":  "crisis",
    "mh":  "neutral",
    "lon": "loneliness",
}

TEXT_COL    = "selftext"
SUB_COL     = "subreddit"
JUNK_VALUES = {"[deleted]", "[removed]", "", "nan", "none"}

# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------
URL_RE          = re.compile(r"https?://\S+|www\.\S+")
MD_LINK_RE      = re.compile(r"\[([^\]]*)\]\([^)]*\)")
SUB_MENTION_RE  = re.compile(r"/?r/\w+")
USER_MENTION_RE = re.compile(r"/?u/\w+")
NON_PRINT_RE    = re.compile(r"[^\x20-\x7E\n]")
WHITESPACE_RE   = re.compile(r"\s+")


def is_junk(text) -> bool:
    return text is None or str(text).strip().lower() in JUNK_VALUES


def clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub(" ", text)
    text = SUB_MENTION_RE.sub(" ", text)
    text = USER_MENTION_RE.sub(" ", text)
    text = NON_PRINT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def infer_label_from_key(s3_key: str) -> Optional[str]:   # Fixed: Optional[str] not str | None
    """Fallback: guess label from the CSV filename prefix."""
    filename = s3_key.split("/")[-1].lower()
    for prefix, label in FILENAME_PREFIX_TO_LABEL.items():
        if filename.startswith(prefix):
            return label
    return None


# --------------------------------------------------------------------------
# S3 helpers
# --------------------------------------------------------------------------
def list_all_csv_keys(s3_client, bucket: str, prefix: str) -> List[str]:   # Fixed: List[str] not list[str]
    """Return every .csv key under prefix using paginated listing."""
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".csv") and not key.startswith("processed/"):
                keys.append(key)
    return sorted(keys)


def read_csv_from_s3(s3_client, bucket: str, key: str) -> Optional[pd.DataFrame]:   # Fixed: Optional not |
    """Download a single S3 object and parse as DataFrame."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        df = pd.read_csv(io.BytesIO(body), on_bad_lines="skip", low_memory=False)
        return df
    except ClientError as e:
        log.warning("S3 read failed for %s: %s", key, e)
        return None
    except Exception as e:
        log.warning("CSV parse failed for %s: %s", key, e)
        return None


def upload_df_to_s3(s3_client, df: pd.DataFrame, bucket: str, key: str) -> None:
    """Serialize DataFrame to CSV in-memory and upload to S3."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    log.info("Uploaded s3://%s/%s  (%d rows)", bucket, key, len(df))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        raise EnvironmentError(
            "AWS credentials not found in environment.\n"
            "Run:\n"
            "  export AWS_ACCESS_KEY_ID=your_access_key\n"
            "  export AWS_SECRET_ACCESS_KEY=your_secret_key\n"
            "  export AWS_DEFAULT_REGION=us-east-1"
        )

    s3 = boto3.client("s3", region_name=AWS_REGION)

    # 1. Discover all CSV keys
    log.info("Listing all CSV files in s3://%s ...", BUCKET)
    all_keys = list_all_csv_keys(s3, BUCKET, RAW_PREFIX)
    log.info("Found %d CSV files", len(all_keys))

    if not all_keys:
        log.error("No CSV files found. Check bucket name and IAM permissions.")
        return

    lookup = {k.lower(): v for k, v in SUBREDDIT_TO_LABEL.items()}

    # 2. Stream + clean + label — one file at a time
    all_chunks    = []
    total_rows    = 0
    kept_rows     = 0
    skipped_files = 0
    label_counts  = Counter()

    for i, key in enumerate(all_keys, 1):
        log.info("[%d/%d] Reading s3://%s/%s", i, len(all_keys), BUCKET, key)

        raw_df = read_csv_from_s3(s3, BUCKET, key)
        if raw_df is None or raw_df.empty:
            skipped_files += 1
            continue

        total_rows += len(raw_df)

        if TEXT_COL not in raw_df.columns:
            log.warning("  No '%s' column — skipping", TEXT_COL)
            skipped_files += 1
            continue

        chunk = raw_df[[TEXT_COL]].copy()

        # Subreddit → label
        if SUB_COL in raw_df.columns:
            normalized = raw_df[SUB_COL].astype(str).str.strip().str.lower()
            chunk["label"] = normalized.map(lookup)
        else:
            chunk["label"] = None

        # Filename fallback
        fallback = infer_label_from_key(key)
        chunk["label"] = chunk["label"].fillna(fallback)
        chunk = chunk.dropna(subset=["label"])

        if chunk.empty:
            log.warning("  No rows with a recognised label — skipping")
            continue

        # Drop junk + clean
        junk_mask = chunk[TEXT_COL].apply(is_junk)
        chunk = chunk.loc[~junk_mask].copy()
        chunk[TEXT_COL] = chunk[TEXT_COL].apply(clean_text)
        chunk = chunk[chunk[TEXT_COL].str.strip().str.len() > 0]

        kept_rows += len(chunk)
        label_counts.update(chunk["label"].value_counts().to_dict())
        all_chunks.append(chunk[[TEXT_COL, "label"]])

        log.info("  Kept %d rows (running total: %d)", len(chunk), kept_rows)

    # 3. Combine + deduplicate
    if not all_chunks:
        log.error("No data extracted — nothing to save.")
        return

    log.info("Combining %d chunks ...", len(all_chunks))
    combined = (
        pd.concat(all_chunks, ignore_index=True)
        .drop_duplicates(subset=[TEXT_COL])
        .reset_index(drop=True)
    )

    # 4. Summary
    log.info("=" * 55)
    log.info("Total raw rows   : %d", total_rows)
    log.info("Rows kept        : %d", kept_rows)
    log.info("Rows after dedup : %d", len(combined))
    log.info("Files skipped    : %d", skipped_files)
    log.info("Label distribution:")
    for label, count in sorted(label_counts.items()):
        log.info("  %-15s %d", label, count)
    log.info("=" * 55)

    # 5. Upload back to S3
    log.info("Uploading to s3://%s/%s ...", BUCKET, OUTPUT_KEY)
    upload_df_to_s3(s3, combined, BUCKET, OUTPUT_KEY)
    log.info("Done.")


if __name__ == "__main__":
    main()