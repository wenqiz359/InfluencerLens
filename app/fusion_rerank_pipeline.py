from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

# =============== basic tools ===============

def dedup_key(item: Dict[str, Any]) -> Tuple:
    return (item.get("doc_id"), item.get("chunk_id", 0))

def text_for_rerank(item: Dict[str, Any]) -> str:
    meta = item.get("metadata") or {}
    tags = " ".join([t for t in meta.get("hashtags", []) if t])[:120]
    author = meta.get("author", "")
    base = item.get("text", "")
    return f"{base}\nAuthor: {author}\nTags: {tags}"

# =============== Step 1: subquery merge + dedup（RRF or Union） ===============

def fuse_across_subqueries(
    results_by_subq: Dict[str, Dict[str, List[Dict[str, Any]]]],
    mode: str = "rrf",                      # "rrf" or "union"
    k_rrf: int = 60,                      
    topk_merge: int = 300,              
    channel_weights: Optional[Dict[str, float]] = None,  # {"vector":1.0,"bm25":1.0}
) -> List[Dict[str, Any]]:
    if channel_weights is None:
        channel_weights = {"vector": 1.0, "bm25": 1.0}

    pool: Dict[Tuple, Dict[str, Any]] = {}

    for subq, by_channel in results_by_subq.items():
        for channel, hits in by_channel.items():
            w = channel_weights.get(channel, 1.0)
            for h in hits:
                key = dedup_key(h)
                if key not in pool:
                    pool[key] = {
                        "id": h["id"],
                        "doc_id": h.get("doc_id"),
                        "chunk_id": h.get("chunk_id", 0),
                        "text": h.get("text", ""),
                        "metadata": h.get("metadata", {}) or {},
                        "scores": {"fusion_score": 0.0},
                        "provenance": set(),
                        "rank": {}
                    }
                item = pool[key]
                item["provenance"].add(channel)

                item.setdefault("per_hits", []).append({
                    "subq": subq,
                    "channel": channel,
                    "rank": h.get("rank", 10**9),
                    "score": h.get("score")
                })

                # merge and score
                r = h.get("rank", 10**9)
                if mode == "rrf":
                    item["scores"]["fusion_score"] += w * (1.0 / (k_rrf + r))
                elif mode == "union":
                    cand = w * (1.0 / (k_rrf + r))
                    item["scores"]["fusion_score"] = max(item["scores"]["fusion_score"], cand)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

    merged = list(pool.values())
    merged.sort(key=lambda x: x["scores"]["fusion_score"], reverse=True)

    for i, it in enumerate(merged, 1):
        it["provenance"] = sorted(list(it["provenance"]))
        it["rank"]["after_fusion"] = i

    return merged[:topk_merge]

# =============== Step 2: candidates to the reranker ===============

def select_for_rerank(
    fused: List[Dict[str, Any]],
    k_rerank: int = 60,            # amount for rerankin（40–80）
    max_per_author: int = 5,      
    min_vec_presence: int = 0   
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    per_author = defaultdict(int)

    for item in fused:
        author = (item.get("metadata") or {}).get("author", "unknown")
        if per_author[author] >= max_per_author:
            continue

        if min_vec_presence > 0:
            if len(item.get("per_hits", [])) < min_vec_presence:
                continue

        out.append(item)
        per_author[author] += 1
        if len(out) >= k_rerank:
            break
    return out

# =============== Step 3: cross encoder reranking ===============

def rerank_with_cross_encoder(
    query: str,
    candidates: List[Dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    topn: int = 10
) -> List[Dict[str, Any]]:
    """
    用 sentence-transformers CrossEncoder 做二次排序，只返回 TopN
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise RuntimeError("Please `pip install sentence-transformers` to use CrossEncoder reranker.")

    if not candidates:
        return []

    reranker = CrossEncoder(model_name)
    pairs = [(query, text_for_rerank(c)) for c in candidates]
    scores = reranker.predict(pairs).tolist()

    for c, s in zip(candidates, scores):
        c.setdefault("scores", {})
        c["scores"]["rerank_score"] = float(s)

    candidates.sort(key=lambda x: x["scores"].get("rerank_score", 0.0), reverse=True)
    for i, it in enumerate(candidates, 1):
        it.setdefault("rank", {})
        it["rank"]["final"] = i

    return candidates[:topn]

# ===============  Pipeline ===============

def build_top10_results(
    query: str,
    results_by_subq: Dict[str, Dict[str, List[Dict[str, Any]]]],
    *,
    fusion_mode: str = "rrf",
    k_rrf: int = 60,
    k_merge: int = 300,
    k_rerank: int = 60,
    max_per_author: int = 5,
    channel_weights: Optional[Dict[str, float]] = None,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    topn: int = 10
) -> List[Dict[str, Any]]:
    """端到端：RRF融合→去重→窄腰→交叉编码器重排→TopN"""
    fused = fuse_across_subqueries(
        results_by_subq,
        mode=fusion_mode,
        k_rrf=k_rrf,
        topk_merge=k_merge,
        channel_weights=channel_weights
    )

    cand = select_for_rerank(
        fused,
        k_rerank=k_rerank,
        max_per_author=max_per_author
    )

    topk = rerank_with_cross_encoder(
        query=query,
        candidates=cand,
        model_name=rerank_model,
        topn=topn
    )
    return topk

# =============== example ===============
# results_by_subq = {
#   "subq_1": {"vector": [ {...} * 20 ], "bm25": [ {...} * 20 ]},
#   "subq_2": {"vector": [ {...} * 20 ], "bm25": [ {...} * 20 ]},
#   # ...
# }
# final_top10 = build_top10_results(
#     query="eco-friendly streetwear for Gen Z brand collab",
#     results_by_subq=results_by_subq,
#     fusion_mode="rrf",
#     k_rrf=60,
#     k_merge=300,
#     k_rerank=60,
#     max_per_author=5,
#     topn=10
# )
# print([ (i["rank"]["final"], i["id"], i["scores"]["rerank_score"]) for i in final_top10 ])
