# -*- coding: utf-8 -*-
"""
Routing agent for merchant queries — parallel parse & rewrite
- Kick off Groq parsing and rewriting concurrently to reduce end-to-end latency.
- Rewriter runs on a lightly "de-numbered" semantic seed to avoid leaking hard filters.
- When parse returns, we adopt its semantic_text + filters, and sanitize already-produced subqueries.

Usage:
  python agent_parse_or_rewrite.py "Find Japanese skincare creators with followers 10k+ and ER > 3%"
  python agent_parse_or_rewrite.py "Glass-skin skincare video ideas and routines" -n 5
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from groq import Groq
from dotenv import load_dotenv

# ----------------------- Config -----------------------
MODEL = "openai/gpt-oss-20b"   # Groq model id (you can swap to another Llama3 on Groq)
load_dotenv()

ALLOWED_FIELDS = sorted([
    "followers", "likes", "comments", "shares", "plays", "engagement_rate",
    "video_duration_sec", "category", "region", "country", "language",
    "gender", "hashtags", "mentions", "post_date", "sponsored"
])

SYSTEM_PARSE = f"""You are a precise query parser for an influencer-retrieval system.
Split the merchant's free-text into: (1) semantic text for dense retrieval and (2) structured hard filters.

Return ONLY valid JSON with this shape:
{{
  "semantic_text": "<string, for embedding; remove hard constraints>",
  "filters": [
    {{
      "field": "<one of: {ALLOWED_FIELDS}>",
      "op": "<=|>=|>|<|=|!=|between|in|not_in|contains|not_contains|exists>",
      "value": "<number|string|bool|list>",
      "unit": "<optional string, e.g., %, followers, sec>",
      "source_span": "<optional short slice from user input>",
      "note": "<optional normalization note>"
    }}
  ],
  "ambiguity_notes": ["<optional>"]
}}

Normalization:
- '1k/5k/10k+' → 1000/5000/10000; '+' => '>='.
- '3%' → engagement_rate 0.03, unit '%'.
- '5–10k followers' → between [5000,10000].
- 'last 30 days' → post_date between ['now-30d','now'].
- Hashtags → hashtags contains <tag>.
- Language/Region/Category → '=' or 'in'.
- Keep 'semantic_text' focused on topical intent (≈10–60 tokens).
"""

USER_TMPL_PARSE = """User input:
{query}

Return the JSON only.
"""

SYSTEM_REWRITE = """You are a professional query rewriter for influencer search.
Expand one natural-language query into several diverse, complementary sub-queries
to maximize recall for a vector search system.

Rules:
- Return a JSON array of strings only.
- Each sub-query 6–18 words, fluent and self-contained.
- Keep the core intent; vary wording, synonyms, tone, audience, or content format.
- Do NOT add numeric filters (follower counts, engagement %) or regions.
- Output ONLY the JSON array, nothing else.
"""

# ----------------------- Groq helpers -----------------------
def _groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("❌ GROQ_API_KEY is not set. Use `export GROQ_API_KEY=...` (macOS/Linux) "
                 "or `setx GROQ_API_KEY ...` (Windows), then restart your terminal.")
    return Groq(api_key=api_key)

def _groq_chat(messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
    client = _groq_client()
    chat = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return chat.choices[0].message.content.strip()

# ----------------------- Text utilities -----------------------
# Patterns for “hard” filters we don't want in semantic sub-queries
HARD_FILTER_RE = re.compile(
    r"""
    (?:\b\d{1,3}(?:,\d{3})+\b)          # 10,000
    |(?:\b\d+(?:\.\d+)?\s*%+\b)         # 3%
    |(?:\b\d+(?:\.\d+)?\s*(?:k|m|K|M)\b) # 10k / 1.2M
    |(?:[<>]=?\s*\d+(?:\.\d+)?)         # >= 1000
    |(?:\blast\s+\d+\s*(?:days?|weeks?|months?)\b)
    |(?:\bbetween\s+\d+\s*(?:k|m|K|M)?\s+and\s+\d+(?:\s*(?:k|m|K|M))?\b)
    |(?:\b(min|max|at\s+least|at\s+most)\b)
    """,
    re.IGNORECASE | re.VERBOSE
)

# Also scrub obvious metric words when tied to numerics (gentle; keeps topical words like “engagement” alone)
METRIC_WORDS = ["followers", "likes", "comments", "shares", "plays", "engagement", "engagement rate"]

def lightly_denumber(text: str) -> str:
    """Remove obvious hard-constraint tokens while preserving topic intent."""
    t = HARD_FILTER_RE.sub("", text)
    # Squash multiple spaces
    t = re.sub(r"\s{2,}", " ", t).strip(" .,:;")
    return t if t else text

def strip_hard_filters_from_subq(s: str) -> str:
    """Post-clean a sub-query: remove numbers/percent/k/m when used as constraints and dangling glue."""
    s2 = HARD_FILTER_RE.sub("", s)
    s2 = re.sub(r"\s{2,}", " ", s2).strip(" .,:;")
    return s2 if s2 else s  # fallback if over-aggressive

def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
      k = x.strip().lower()
      if k and k not in seen:
          out.append(x.strip())
          seen.add(k)
    return out

# ----------------------- Heuristic (still useful for telemetry/route tag) -----------------------
FILTER_PATTERNS = [
    r"\b\d{1,3}(?:,\d{3})+\b",
    r"\b\d+(?:\.\d+)?\s*%+",
    r"\b\d+(?:\.\d+)?\s*(k|K|m|M)\b",
    r"[<>]=?\s*\d+",
    r"\bbetween\s+\d+\s*(?:k|K|m|M)?\s+and\s+\d+",
    r"\b(min|max|at least|at most)\b",
    r"#\w+",
    r"\blast\s+\d+\s*(days|day|weeks|week|months|month)\b",
    r"\bnon[-\s]?sponsored|\bsponsored\b",
    r"\bfollower|followers|likes|comments|shares|plays|engagement\b",
]

def looks_like_has_filters(text: str, threshold: int = 1) -> bool:
    hits = sum(1 for p in FILTER_PATTERNS if re.search(p, text, flags=re.IGNORECASE))
    return hits >= threshold

# ----------------------- Parse & Rewrite (unchanged LLM prompts) -----------------------
def parse_merchant_input(text: str) -> Dict[str, Any]:
    raw = _groq_chat(
        messages=[
            {"role": "system", "content": SYSTEM_PARSE},
            {"role": "user", "content": USER_TMPL_PARSE.format(query=text)},
        ],
        temperature=0.1,  # deterministic parsing
    )
    try:
        obj = json.loads(raw)
        assert isinstance(obj, dict) and "semantic_text" in obj and "filters" in obj
        return obj
    except Exception:
        m = re.search(r"\{.*\}\s*$", raw, re.S)
        if not m:
            raise ValueError(f"Parser did not return JSON:\n{raw}")
        return json.loads(m.group(0))

def rewrite_semantic_text(semantic_text: str, n: int = 5, temperature: float = 0.8) -> List[str]:
    user = f'Original query:\n"{semantic_text}"\n\nPlease generate {n} diverse sub-queries.'
    raw = _groq_chat(
        messages=[
            {"role": "system", "content": SYSTEM_REWRITE},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    try:
        subs = json.loads(raw)
        if not isinstance(subs, list):
            raise ValueError("Not a list")
    except Exception:
        subs = [s.strip("-• ").strip() for s in raw.splitlines() if s.strip()]
    # Ensure we include the seed as first item
    out, seen = [], set()
    seed = semantic_text.strip()
    if seed:
        out.append(seed); seen.add(seed)
    for s in subs:
        s = s.strip()
        if s and s not in seen:
            out.append(s); seen.add(s)
    return out[: n + 1]

# ----------------------- Parallel Router -----------------------
def route_and_run_parallel(text: str, n_rewrites: int, temp: float) -> Dict[str, Any]:
    """
    Parallel policy:
      - Fire PARSE(text) and REWRITE(lightly_denumber(text)) concurrently.
      - When both return, choose:
          semantic_text := parsed.semantic_text (if present) else lightly_denumber(text)
          filters       := parsed.filters (or [])
          subqueries    := post-clean the already-produced subqueries to strip hard filters.
      - Route tag is for observability only.
    """
    route_tag = "unknown"
    try:
        route_tag = "likely_has_filters" if looks_like_has_filters(text) else "likely_no_filters"
    except Exception:
        pass

    seed_for_rewrite = lightly_denumber(text)

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_parse = ex.submit(parse_merchant_input, text)
        fut_rewr  = ex.submit(rewrite_semantic_text, seed_for_rewrite, n_rewrites, temp)

        parsed, subqs = None, None
        for fut in as_completed([fut_parse, fut_rewr]):
            # We just spin until both are done; exceptions will surface here.
            pass
        parsed = fut_parse.result()
        subqs  = fut_rewr.result()

    # Adopt best semantic_text from parse, fallback to our denumbered seed
    semantic_text = (parsed.get("semantic_text") or seed_for_rewrite).strip()
    filters = parsed.get("filters", [])
    notes = parsed.get("ambiguity_notes", [])

    # Post-sanitize subqueries (remove hard numbers/k/M/%/ranges that slipped in)
    cleaned = [strip_hard_filters_from_subq(s) for s in subqs]
    cleaned = [s for s in cleaned if s]  # drop empties
    cleaned = dedup_keep_order(cleaned)

    # Ensure first one equals final semantic_text (for consistency downstream)
    if not cleaned or cleaned[0] != semantic_text:
        cleaned = dedup_keep_order([semantic_text] + cleaned)

    return {
        "route": route_tag + " | parallel_parse_and_rewrite",
        "semantic_text": semantic_text,
        "filters": filters,
        "ambiguity_notes": notes,
        "subqueries": cleaned  # semantic seed + rewrites, ready for per-subquery embedding
    }

# ----------------------- CLI -----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent router (parallel): parse & rewrite concurrently via Groq.")
    parser.add_argument("text", nargs="?", help="Merchant input (may include filters).")
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of rewrites (default 5).")
    parser.add_argument("--temp", type=float, default=0.8, help="Rewrite temperature (default 0.8).")
    args = parser.parse_args()

    text = args.text or input("Enter merchant input: ").strip()
    if not text:
        sys.exit("❌ No input provided.")

    result = route_and_run_parallel(text, n_rewrites=args.num, temp=args.temp)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

