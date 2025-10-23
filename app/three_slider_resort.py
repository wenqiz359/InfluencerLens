"""
===========================================================
Function: resort_topk_with_three_weights
-----------------------------------------------------------
Purpose:
    This function performs the final weighted re-ranking of 
    candidate influencers after the cross-encoder reranker step.

    It combines three high-level user-adjustable priorities:
        1. relevance        → based on cross-encoder score (semantic relevance)
        2. followers        → based on normalized follower count
        3. user_interaction → internally weighted combination of likes, comments,
                              plays, and shares

    Each slider (0–1) controls the relative importance of these three aspects. 
    Internally, raw metrics are normalized (log1p + min–max) to reduce skew from 
    heavy-tailed distributions (e.g., follower counts), then combined using a 
    weighted sum. The final score determines the display ranking on the frontend.

Typical Usage:
    1. Run `rerank_with_cross_encoder()` to obtain top-k candidates with `rerank_score`.
    2. Call this function to re-rank those candidates based on user sliders:
         final_top10 = resort_topk_with_three_weights(
             candidates=reranked_top20,
             sliders={"relevance": 0.9, "followers": 0.4, "user_interaction": 0.7},
             topn=10
         )
    3. (Optional) Pass the final ranked list to an LLM explanation layer to generate
       "Why Matched" justifications for each result.

Outputs:
    - Adds these normalized metrics to each candidate:
        scores.relevance_norm
        scores.followers_norm
        scores.ui_internal_score
        scores.final_score
    - Updates `rank.final_priority` to reflect the new order.
    - Returns the top N candidates (default = 10).

Integration in pipeline:
    retrieve_for_subqueries → fuse_across_subqueries → select_for_rerank →
    rerank_with_cross_encoder → resort_topk_with_three_weights → generate_why_matched
===========================================================
"""

from typing import List, Dict, Any
import math

# === 可按需要改成你的 metadata 字段名 ===
UI_FIELDS = {
    "likes":     "likes",
    "comments":  "comments",
    "plays":     "avg_plays",   # 或 "play_median"
    "shares":    "shares",
}
FOLLOWERS_FIELD = "followers"

def _log_minmax(xs: List[float]) -> List[float]:
    ys = [math.log1p(max(0.0, x)) for x in xs]
    lo, hi = min(ys), max(ys)
    if hi <= lo: return [0.5] * len(xs)
    return [(y - lo) / (hi - lo) for y in ys]

def _lin_minmax(xs: List[float]) -> List[float]:
    lo, hi = min(xs), max(xs)
    if hi <= lo: return [0.5] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]

def _get(meta: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(meta.get(key, default))
    except Exception:
        return default

def resort_topk_with_three_weights(
    candidates: List[Dict[str, Any]],
    *,
    # 三条滑杆（UI 0~1）：relevance / followers / user_interaction
    sliders: Dict[str, float],           # e.g. {"relevance":0.9,"followers":0.4,"user_interaction":0.7}
    # user_interaction 的内部权重（工程内定，给 PM 调）
    ui_internal_weights: Dict[str, float] = None,  # e.g. {"likes":0.4,"comments":0.3,"plays":0.2,"shares":0.1}
    topn: int = 10,
    amplify_pow: float = 2.0             # 放大滑杆差异：w -> w**pow
) -> List[Dict[str, Any]]:
    """
    假设 candidates 里已有 scores.rerank_score（来自 cross-encoder）。
    在这批候选内部做一次融合排序，返回前 topn。
    """
    if not candidates:
        return []

    if ui_internal_weights is None:
        ui_internal_weights = {"likes": 0.35, "comments": 0.30, "plays": 0.25, "shares": 0.10}

    metas = [c.get("metadata") or {} for c in candidates]

    # 1) 取原始值
    followers_raw = [_get(m, FOLLOWERS_FIELD, 0.0) for m in metas]
    ui_raw = {k: [_get(m, UI_FIELDS[k], 0.0) for m in metas] for k in UI_FIELDS}

    # 2) 归一化：长尾计数走 log1p+minmax；rerank 线性 minmax
    followers_norm = _log_minmax(followers_raw)
    ui_norm_each = {k: _log_minmax(vs) for k, vs in ui_raw.items()}
    reraw = [float(c.get("scores", {}).get("rerank_score", 0.0)) for c in candidates]
    rel_norm = _lin_minmax(reraw)

    # 3) 先把四个互动维度内部融合成一个 user_interaction_norm ∈ [0,1]
    #    这里直接线性加权；也可改成几何平均等
    ui_score = [0.0] * len(candidates)
    for name, w in ui_internal_weights.items():
        for i, v in enumerate(ui_norm_each[name]):
            ui_score[i] += float(w) * v

    # 4) 三条滑杆权重（放大使差异更明显）
    def W(name: str) -> float:
        return max(0.0, float(sliders.get(name, 0.0))) ** amplify_pow
    w_rel = W("relevance")
    w_fol = W("followers")
    w_ui  = W("user_interaction")

    # 5) 计算最终分并排序
    for i, c in enumerate(candidates):
        biz_followers = w_fol * followers_norm[i]
        biz_ui        = w_ui  * ui_score[i]
        final = w_rel * rel_norm[i] + biz_followers + biz_ui

        c.setdefault("scores", {})
        c["scores"]["relevance_norm"]   = rel_norm[i]
        c["scores"]["followers_norm"]   = followers_norm[i]
        c["scores"]["ui_internal_score"] = ui_score[i]
        c["scores"]["final_score"]      = float(final)

    candidates.sort(key=lambda x: x["scores"]["final_score"], reverse=True)
    for j, it in enumerate(candidates, 1):
        it.setdefault("rank", {})
        it["rank"]["final_priority"] = j
    return candidates[:topn]
