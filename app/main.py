"""
main.py — Unified RAG-style influencer retrieval pipeline (Neo4j-only)

Flow:
1) Parse or rewrite merchant input (parse_or_rewrite.py)
2) Retrieve candidates from Neo4j: BM25 (full-text over combined_text+summary) + Vector (embedding_summary)
3) Fuse all subqueries and channels via Reciprocal Rank Fusion
4) Rerank top candidates with a cross-encoder
5) Re-sort by user-defined sliders (relevance / followers / interaction)
6) Generate short "why matched" explanations with LLM
"""

# ================= Imports =================
import json
import os
import time
from typing import List
from neo4j import GraphDatabase
from openai import OpenAI
from dotenv import load_dotenv

# --- Internal modules you already wrote ---
from parse_or_rewrite import route_and_run_parallel as route_and_run
from retrieve_dual import retrieve_for_subqueries            
from fusion_rerank_pipeline import (
    fuse_across_subqueries,
    select_for_rerank,
    rerank_with_cross_encoder
)
from three_slider_resort import resort_topk_with_three_weights
from explain_why_match import generate_why_matched_with_groq


# ================= Config via .env =================
load_dotenv()

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")

# Index names (match init_neo4j.cypher)
NEO4J_FT_INDEX = os.getenv("NEO4J_FT_INDEX", "influencer_ft")               # BM25 full-text index
NEO4J_VECTOR_INDEX = os.getenv("NEO4J_VEC_INDEX", "influencer_summary_vec")  # Vector index

# Cross-encoder model
CROSS_ENCODER_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Embedding API (OpenAI-compatible; you用的是 DashScope 网关)
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v4")
EMBED_DIM = int(os.getenv("VEC_DIM", "1024"))

if not QWEN_API_KEY:
    raise ValueError("❌ Please set QWEN_API_KEY in your .env")

client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)


# ---------------- Embedding ----------------
def embed_fn(
    text_or_list,
    model: str = EMBED_MODEL,
    dimensions: int = EMBED_DIM,
    retry: int = 3,
    sleep_sec: float = 2.0,
):
    """
    Generate embeddings via an OpenAI-compatible endpoint.
    - Accepts str or List[str]; returns List[float] for str, or List[List[float]] for list.
    - 'dimensions' must match your Neo4j vector index dim.
    """
    if isinstance(text_or_list, str):
        batch = [text_or_list] if text_or_list else []
        single = True
    else:
        batch = [t for t in text_or_list if t]  # filter empties
        single = False

    if not batch:
        zeros = [0.0] * dimensions
        return zeros if single else [[0.0] * dimensions for _ in range(len(text_or_list))]

    last_err = None
    for attempt in range(1, retry + 1):
        try:
            resp = client.embeddings.create(
                model=model,
                input=batch,
                dimensions=dimensions,
            )
            vecs = [d.embedding for d in resp.data]
            return vecs[0] if single else vecs
        except Exception as e:
            last_err = e
            print(f"⚠️ Embedding attempt {attempt}/{retry} failed: {e}")
            time.sleep(sleep_sec * attempt)

    raise RuntimeError(f"❌ Embedding failed after {retry} attempts: {last_err}")


# ================= Pipeline =================
def main():
    # 1) Merchant input & query routing
    merchant_query = input("🧾 Enter merchant query: ").strip()
    if not merchant_query:
        print("❌ Empty input.")
        return

    print("\n🤖 Routing & rewriting query ...")
    parsed = route_and_run(merchant_query, n_rewrites=5, temp=0.8)
    subqueries: List[str] = parsed["subqueries"]
    filters = parsed.get("filters") or {}
    print(f"✅ Generated {len(subqueries)} sub-queries:")
    for s in subqueries:
        print("   •", s)
    if filters:
        print("📊 Detected structured filters:", json.dumps(filters, ensure_ascii=False, indent=2))

    # 2) Retrieval from Neo4j only (BM25 via full-text + Vector)
    print("\n🔍 Retrieving from Neo4j (BM25 + Vector) ...")
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    results_by_subq = retrieve_for_subqueries(
        subqueries=subqueries,
        neo4j_driver=neo4j_driver,
        neo4j_fulltext_index=NEO4J_FT_INDEX,
        neo4j_vector_index=NEO4J_VECTOR_INDEX,
        embed_fn=lambda x: embed_fn(x, model=EMBED_MODEL, dimensions=EMBED_DIM),
        k_each=20,
    )

    print("✅ Retrieval complete. Example counts per subquery:")
    print(json.dumps({k: {ch: len(v) for ch, v in d.items()} for k, d in results_by_subq.items()}, indent=2))

    # 3) Fuse across subqueries (RRF)
    print("\n⚙️ Fusing results (RRF) ...")
    fused = fuse_across_subqueries(results_by_subq, mode="rrf", k_rrf=60, topk_merge=300)

    # 4) Candidate narrowing + rerank
    print("🧮 Selecting candidates for rerank ...")
    cand = select_for_rerank(fused, k_rerank=60, max_per_author=5)

    print("💡 Cross-encoder reranking ...")
    reranked = rerank_with_cross_encoder(
        query=parsed.get("semantic_text") or merchant_query,
        candidates=cand,
        model_name=CROSS_ENCODER_MODEL,
        topn=20
    )

    # 5) Weighted priority sort (UI sliders)
    sliders = {"relevance": 0.9, "followers": 0.4, "user_interaction": 0.7}
    print(f"🎚️ Re-sorting by sliders {sliders} ...")
    final_top10 = resort_topk_with_three_weights(
        candidates=reranked,
        sliders=sliders,
        topn=10
    )

    # 6) LLM “Why Matched” explanations
    print("\n💬 Generating 'why matched' explanations ...")
    final_top10 = generate_why_matched_with_groq(
        query=parsed.get("semantic_text") or merchant_query,
        candidates=final_top10,
        sliders=sliders
    )

    # 7) Show results
    print("\n================= 🏁 FINAL TOP RESULTS =================")
    for i, item in enumerate(final_top10, 1):
        meta = item.get("metadata", {})
        author = meta.get("author", "Unknown")
        followers = meta.get("followers")
        likes = meta.get("likes") or meta.get("likes_avg")
        why = item.get("why_matched", "")
        text = (item.get("text", "") or "")[:160]

        print(f"\n#{i}: {author}")
        print("Followers:", followers, "| Likes:", likes)
        print("Text:", text + ("..." if len(text) == 160 else ""))
        print("Why matched:", why)
        print("Final score:", round(item.get("scores", {}).get("final_score", 0.0), 4))

    print("\n✅ Pipeline complete.\n")


# ================= Entry Point =================
if __name__ == "__main__":
    main()
