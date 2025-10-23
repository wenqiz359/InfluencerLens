#!/usr/bin/env python3
"""
Data Preprocessing for AC215 - TikTok Influencer Analysis
Converted from Jupyter notebook to Python script
"""

# Install required packages (uncomment if needed)
# pip install orjson pandas numpy tqdm regex langdetect sentence-transformers torch emoji groq openai

import os, re, math, time, emoji
import numpy as np
import orjson
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

# ---------------- Load Environment ----------------
load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GCS_BUCKET = os.getenv("GCS_BUCKET")
RAW_BLOB = os.getenv("RAW_BLOB")
OUTPUT_BLOB = os.getenv("OUTPUT_BLOB")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")


if not QWEN_API_KEY or not QWEN_BASE_URL:
    raise ValueError("Missing Qwen API credentials in .env!")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env!")

qwen_client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- Google Cloud Storage I/O ----------------
from google.cloud import storage

def download_from_gcs(bucket_name, src_blob, dest_path):
    from google.cloud import storage
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(src_blob)
    blob.download_to_filename(dest_path)
    print(f"Downloaded gs://{bucket_name}/{src_blob} → {dest_path}")

def upload_to_gcs(bucket_name, src_path, dest_blob):
    from google.cloud import storage
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(src_path)
    print(f"Uploaded {src_path} → gs://{bucket_name}/{dest_blob}")


# ---------------- Regex & Utils ----------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
HASHTAG_SYMBOL_RE = re.compile(r"#")

def clean_text(s, remove_hashtags=False):
    if not s:
        return ""
    s = s.lower()
    s = emoji.demojize(s, language="en")
    s = re.sub(r":([a-z_]+):", r"\1", s)
    s = URL_RE.sub(" ", s)
    if remove_hashtags:
        s = re.sub(r"#\w+", " ", s)
    s = HASHTAG_SYMBOL_RE.sub("", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s

def dedup_preserve_order(texts):
    seen, out = set(), []
    for t in texts:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out

def safe_parse_time(ts):
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def isoformat_utc(dt):
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def safe_mean(vals):
    arr = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(arr)) if arr else None

# ---------------- Batch Embedding ----------------
def batch_embed(texts: List[str], model: str = "text-embedding-v4") -> List[List[float]]:
    """Batch embedding using text-embedding-v4 API."""
    results = []
    BATCH_SIZE = 10
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating embeddings"):
        batch = texts[i:i+BATCH_SIZE]
        try:
            resp = qwen_client.embeddings.create(model=model, input=batch)
            results.extend([d.embedding for d in resp.data])
        except Exception as e:
            print(f"Embedding batch {i//BATCH_SIZE} failed: {e}")
            results.extend([[0.0]*1024 for _ in batch])
        time.sleep(0.3)
    return results

# ---------------- True Batch Summarization ----------------
def batch_summarize(texts: List[str], batch_size: int = 10) -> List[str]:
    results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_id = i // batch_size + 1
        print(f"[Batch {batch_id}/{total_batches}] ⏳ Generating summaries...", flush=True)

        batch = texts[i:i+batch_size]

        # 编号拼接输入
        numbered_texts = "\n\n".join(
            [f"[INFLUENCER {idx+1}]\n{text}" for idx, text in enumerate(batch)]
        )

        prompt = f"""
You are an NLP assistant performing semantic analysis of TikTok influencers' content.

Below are multiple influencers' combined texts (hashtags + bio + posts).
For each influencer, generate a concise and professional summary describing their key topics, tone, and style.
Do NOT include numbers or metrics. Capture all relevant hashtags, bio, and post text faithfully.

Return the output in this format:

[INFLUENCER 1 SUMMARY]: ...
[INFLUENCER 2 SUMMARY]: ...
...

Combined texts:
{numbered_texts}
"""

        try:
            resp = openai_client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}]
            )
            content = resp.choices[0].message.content.strip()

            # 按编号解析输出
            batch_results = []
            for j in range(len(batch)):
                pattern = re.compile(
                    rf"\[INFLUENCER {j+1} SUMMARY\]:(.*?)(?=\[INFLUENCER {j+2} SUMMARY\]|$)",
                    re.S
                )
                match = pattern.search(content)
                if match:
                    batch_results.append(match.group(1).strip())
                else:
                    batch_results.append("")
            results.extend(batch_results)
            print(f"[Batch {batch_id}] ✅ Done.", flush=True)

        except Exception as e:
            print(f"⚠️ GPT batch {batch_id} failed: {e}", flush=True)
            results.extend([""] * len(batch))
        time.sleep(0.5)

    return results

# ---------------- Stage 1: Preprocess influencers ----------------
INFLUENCER_DROP_FIELDS = {
    "id", "bioLink", "verified", "originalAvatarUrl", "avatar",
    "commerceUserInfo", "ttSeller", "privateAccount",
    "friends", "following", "roomId", "heart", "digg"
}

POST_DROP_FIELDS = {
    "id", "mentions", "musicMeta", "isAd", "isSponsored",
    "originalCoverUrl", "coverUrl", "videoMeta"
}

def preprocess_influencers(raw):
    processed = []
    for inf in tqdm(raw, desc="Building combined_text"):
        posts = inf.get("posts", [])
        if not posts:
            continue

        clean_inf = {k: v for k, v in inf.items() if k not in INFLUENCER_DROP_FIELDS}
        name = clean_inf.get("name") or clean_inf.get("nickName")
        profile_url = clean_inf.get("profileUrl")
        bio_block = clean_text(clean_inf.get("signature", ""))

        texts, hashtags_all = [], []
        diggs, comments, shares, plays, collects, durations = [], [], [], [], [], []
        latest_dt = None

        for p in posts:
            p = {k: v for k, v in p.items() if k not in POST_DROP_FIELDS}

            txt = clean_text(p.get("text"), remove_hashtags=True)
            if txt:
                texts.append(txt)
            tags = p.get("hashtags", [])
            for t in tags:
                if isinstance(t, dict) and "name" in t:
                    hashtags_all.append(clean_text(t["name"]))
                elif isinstance(t, str):
                    hashtags_all.append(clean_text(t.lstrip("#")))

            for key, arr in zip(
                ["diggCount", "commentCount", "shareCount", "playCount", "collectCount"],
                [diggs, comments, shares, plays, collects]
            ):
                v = p.get(key)
                try:
                    arr.append(float(v))
                except:
                    pass

            try:
                durations.append(float(p.get("videoDuration", np.nan)))
            except:
                pass

            dt = safe_parse_time(p.get("createTime"))
            if dt and (latest_dt is None or dt > latest_dt):
                latest_dt = dt

        text_block = " ".join(dedup_preserve_order(texts)).strip()
        hashtags_block = " ".join(dedup_preserve_order(hashtags_all))
        if not text_block and not hashtags_block and not bio_block:
            continue

        combined_text = f"Hashtags: {hashtags_block}. Bio: {bio_block}. Posts: {text_block}"

        processed.append({
            "name": name,
            "combined_text": combined_text,
            "metadata": {
                "author": name,
                "followers": clean_inf.get("fans"),
                "signature": bio_block,
                "likes_avg": safe_mean(diggs),
                "comments_avg": safe_mean(comments),
                "shares_avg": safe_mean(shares),
                "plays_avg": safe_mean(plays),
                "collects_avg": safe_mean(collects),
                "avg_video_duration_sec": safe_mean(durations),
                "hashtags": dedup_preserve_order(hashtags_all),
                "language": "en",
                "created_at": isoformat_utc(latest_dt),
                "source_url": profile_url
            }
        })
    return processed

# ---------------- Main ----------------
def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    local_raw = Path("/tmp/final_merge.json")
    local_out = Path("/shared/influencer_index.jsonl")
    log_path = Path("/shared/error.log")

    try:
        print("Step 1: Downloading from GCS ...")
        download_from_gcs(GCS_BUCKET, RAW_BLOB, local_raw)

        print("Step 2: Preprocessing influencers ...")
        raw = orjson.loads(local_raw.read_bytes())
        influencers = raw if isinstance(raw, list) else [raw]
        processed = preprocess_influencers(influencers)

        print("Step 3: Generating summaries ...")
        texts = [p["combined_text"] for p in processed]
        summaries = batch_summarize(texts, batch_size=8)

        print("Step 4: Generating embeddings ...")
        embeddings = batch_embed(summaries)

        print("Step 5: Saving output to shared folder ...")
        for p, summary, emb in zip(processed, summaries, embeddings):
            p["summary"] = summary
            p["embedding_summary"] = emb

        local_out.write_bytes(b"\n".join([orjson.dumps(rec) for rec in processed]))
        print(f"JSONL saved to shared volume at {local_out}")

    except Exception as e:
        msg = f"Pipeline failed: {e}"
        print(msg)
        log_path.write_text(msg)
        raise

if __name__ == "__main__":
    main()