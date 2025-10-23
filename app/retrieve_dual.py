# retrieve_dual.py
# Neo4j-only dual retrieval:
# - BM25 via Full-Text over (combined_text + summary)
# - Vector KNN over embedding_summary
# Returns results in the normalized shape expected by your fusion/rerank pipeline.

from __future__ import annotations
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import os

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# =========================
# 0) Config & Driver helper
# =========================

@dataclass
class Neo4jConf:
    # 默认指向 docker-compose 内部服务名，可被 .env 覆盖
    uri: str = os.getenv("NEO4J_URI", "bolt://neo4j-db:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    ft_index: str = os.getenv("NEO4J_FT_INDEX", "influencer_ft")
    vec_index: str = os.getenv("NEO4J_VEC_INDEX", "influencer_summary_vec")

def get_neo4j_driver(conf: Neo4jConf | None = None):
    conf = conf or Neo4jConf()
    return GraphDatabase.driver(conf.uri, auth=(conf.user, conf.password))

# =========================
# 1) Normalization schema
# =========================

def unify_item(
    *,
    source: str,                # "bm25" | "vector"
    id_: str,
    doc_id: str,
    chunk_id: int,
    text: str,
    metadata: Dict[str, Any],
    score: float,
    rank: int,
    index_name: str,
    highlights: Optional[Dict[str, List[str]]] = None,
    raw: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Return a normalized candidate item for downstream fusion/rerank."""
    return {
        "id": id_,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text,
        "metadata": metadata or {},
        "score": float(score),     # Higher is better
        "rank": int(rank),         # Rank within this channel result set (1-based)
        "source": source,          # "bm25" or "vector"
        "index": index_name,
        "highlights": highlights,
        "raw": raw or {}
    }

# =========================
# 1.5) Lucene helpers
# =========================

def _lucene_escape(q: str) -> str:
    # Escape Lucene special chars: + - ! ( ) { } [ ] ^ " ~ * ? : \ /
    chars = r'+-!(){}[]^"~*?:\\/'
    out = []
    for ch in q:
        out.append("\\" + ch if ch in chars else ch)
    return "".join(out)

def _build_lucene_query_both_fields(user_q: str) -> str:
    """
    同时在 combined_text 与 summary 两个字段做 BM25 检索，等权。
    形如：(combined_text:(...)) OR (summary:(...))
    """
    q = _lucene_escape((user_q or "").strip())
    if not q:
        return ""  # 避免空查询
    return f"(combined_text:({q})) OR (summary:({q}))"

# ===================================
# 2) Neo4j Full-Text (BM25) retrieval
# ===================================

def neo4j_fulltext_search_subquery(
    driver,
    fulltext_index_name: str,
    subquery: str,
    k: int = 20,
) -> List[Dict[str, Any]]:
    """
    用 Neo4j full-text (Lucene/BM25) 同时检索 combined_text + summary。
    兼容 row['node'] 为 Node 或 dict 的两种情况。
    """
    lucene_q = _build_lucene_query_both_fields(subquery)

    cypher = """
    CALL db.index.fulltext.queryNodes($ft_idx, $q, {limit: $k})
    YIELD node, score
    RETURN elementId(node) AS element_id,
           labels(node)    AS labels,
           node            AS node,
           score
    ORDER BY score DESC
    LIMIT $k
    """

    with driver.session() as sess:
        rows = sess.run(cypher, ft_idx=fulltext_index_name, q=lucene_q, k=k).data()

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        node_obj = row.get("node")

        # 兼容 Neo4j Node / 已序列化 dict
        if isinstance(node_obj, dict):
            props = node_obj
            neo4j_element_id = row.get("element_id") or node_obj.get("element_id")
        else:
            try:
                props = dict(node_obj)
            except Exception:
                props = {}
            neo4j_element_id = row.get("element_id") or getattr(node_obj, "element_id", None)

        score = float(row.get("score", 0.0))

        # 字段映射（按你的 schema）
        name = props.get("name") or ""
        text = props.get("combined_text") or props.get("summary") or ""

        metadata = {
            "author": name,
            "followers": props.get("followers"),
            "likes_avg": props.get("likes_avg"),
            "comments_avg": props.get("comments_avg"),
            "shares_avg": props.get("shares_avg"),
            "plays_avg": props.get("plays_avg"),
            "collects_avg": props.get("collects_avg"),
            "avg_video_duration_sec": props.get("avg_video_duration_sec"),
            "source_url": props.get("source_url"),
            "labels": row.get("labels") or [],
        }

        out.append(unify_item(
            source="bm25",
            id_=neo4j_element_id or name,
            doc_id=name or (neo4j_element_id or ""),
            chunk_id=0,
            text=text,
            metadata=metadata,
            score=score,
            rank=i,
            index_name=fulltext_index_name,
            highlights=None,
            raw={"neo4j_element_id": neo4j_element_id, "bm25": score, "node": props, "lucene_q": lucene_q}
        ))
    return out

# ==============================
# 3) Neo4j Vector (KNN) retrieval
# ==============================

def neo4j_vector_search_subquery(
    driver: GraphDatabase.driver,
    vector_index_name: str,
    subquery: str,
    *,
    embed_fn: Callable[[str], List[float]],
    k: int = 20,
    metric: str = "cosine"  # "cosine" | "dot" | "euclidean"
) -> List[Dict[str, Any]]:

    # 1) 生成查询向量（维度需与索引一致）
    qvec = embed_fn(subquery)

    # 2) KNN 检索（把 elementId/labels 一起带回，兼容 Node/Dict）
    cypher = """
    CALL db.index.vector.queryNodes($index, $k, $embedding)
    YIELD node, score
    RETURN elementId(node) AS element_id,
           labels(node)    AS labels,
           node            AS node,
           score
    ORDER BY score DESC
    LIMIT $k
    """

    with driver.session() as sess:
        rows = sess.run(cypher, index=vector_index_name, k=k, embedding=qvec).data()

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        node_obj = row.get("node")

        # 兼容 Neo4j Node / 已序列化 dict
        if isinstance(node_obj, dict):
            props = node_obj
            neo4j_element_id = row.get("element_id") or node_obj.get("element_id")
        else:
            try:
                props = dict(node_obj)  # neo4j.types.graph.Node
            except Exception:
                props = {}
            neo4j_element_id = row.get("element_id") or getattr(node_obj, "element_id", None)

        raw_score = float(row.get("score", 0.0))

        # 3) 不同度量归一化为“越大越好”
        if metric.lower() == "euclidean":
            # Neo4j 有时返回负距离；统一转为正距离
            dist = abs(raw_score)
            score = 1.0 / (1.0 + dist)  # 0~1，越近越大
        else:
            score = raw_score           # cosine/dot 本身是相似度

        # 4) 字段映射（按你的 schema）
        name = props.get("name") or ""
        text = props.get("summary") or props.get("combined_text") or ""

        metadata = {
            "author": name,
            "followers": props.get("followers"),
            "likes_avg": props.get("likes_avg"),
            "comments_avg": props.get("comments_avg"),
            "shares_avg": props.get("shares_avg"),
            "plays_avg": props.get("plays_avg"),
            "collects_avg": props.get("collects_avg"),
            "avg_video_duration_sec": props.get("avg_video_duration_sec"),
            "source_url": props.get("source_url"),
            "labels": row.get("labels") or [],
        }

        out.append(unify_item(
            source="vector",
            id_=neo4j_element_id or name,
            doc_id=name or (neo4j_element_id or ""),
            chunk_id=0,
            text=text,
            metadata=metadata,
            score=score,
            rank=i,
            index_name=vector_index_name,
            highlights=None,
            raw={
                "neo4j_element_id": neo4j_element_id,
                "sim_or_dist": raw_score,
                "metric": metric,
                "node": props
            }
        ))
    return out

# ========================================
# 4) High-level per-subquery & multi-subq
# ========================================

def retrieve_for_subquery(
    subquery: str,
    *,
    neo4j_driver: GraphDatabase.driver,
    neo4j_fulltext_index: str,
    neo4j_vector_index: str,
    embed_fn: Callable[[str], List[float]],
    k_each: int = 20,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return {"vector": [...], "bm25": [...]} for one subquery.
    """
    bm25_hits = neo4j_fulltext_search_subquery(
        driver=neo4j_driver,
        fulltext_index_name=neo4j_fulltext_index,
        subquery=subquery,
        k=k_each,
    )
    vector_hits = neo4j_vector_search_subquery(
        driver=neo4j_driver,
        vector_index_name=neo4j_vector_index,
        subquery=subquery,
        embed_fn=embed_fn,
        k=k_each,
        metric="cosine",
    )
    return {"vector": vector_hits, "bm25": bm25_hits}

def retrieve_for_subqueries(
    subqueries: List[str],
    *,
    neo4j_driver: GraphDatabase.driver,
    neo4j_fulltext_index: str,
    neo4j_vector_index: str,
    embed_fn: Callable[[str], List[float]],
    k_each: int = 20,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    For a list of subqueries, return:
      {
        "subq_1": {"vector": [...], "bm25": [...]},
        "subq_2": {"vector": [...], "bm25": [...]},
        ...
      }
    """
    results_by_subq: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for idx, sq in enumerate(subqueries, 1):
        key = f"subq_{idx}"
        results_by_subq[key] = retrieve_for_subquery(
            subquery=sq,
            neo4j_driver=neo4j_driver,
            neo4j_fulltext_index=neo4j_fulltext_index,
            neo4j_vector_index=neo4j_vector_index,
            embed_fn=embed_fn,
            k_each=k_each,
        )
    return results_by_subq

# =========================
# 5) Example runnable usage
# =========================
if __name__ == "__main__":
    # 1) Neo4j driver
    conf = Neo4jConf()
    driver = get_neo4j_driver(conf)

    # 2) Embedding fn (placeholder) — return list[float] of size VEC_DIM
    # Replace with your real embedder (Qwen/BGE etc.)
    VEC_DIM = int(os.getenv("VEC_DIM", "1024"))
    def embed_fn(text: str) -> List[float]:
        # TODO: plug real model. For now, return a zero vector for smoke test.
        return [0.0] * VEC_DIM

    # 3) Example subqueries
    subqs = [
        "korean sweet style influencer",
        "eco friendly streetwear",
        "minimalist outfit inspiration"
    ]

    results = retrieve_for_subqueries(
        subqueries=subqs,
        neo4j_driver=driver,
        neo4j_fulltext_index=conf.ft_index,
        neo4j_vector_index=conf.vec_index,
        embed_fn=embed_fn,
        k_each=20,
    )

    # Shape check for your fusion:
    print({k: {ch: len(v) for ch, v in chs.items()} for k, chs in results.items()})
