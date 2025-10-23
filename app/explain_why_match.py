from typing import List, Dict, Any
from groq import Groq 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize Groq client
client = OpenAI(api_key="OPENAI_API_KEY")

def generate_why_matched_with_groq(
    query: str,
    candidates: List[Dict[str, Any]],
    sliders: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Use  LLM to generate a concise 'why matched' explanation
    for each ranked candidate.
    
    Args:
        query: The brand or search query.
        candidates: Ranked influencer list (e.g., output from three-slider ranking).
        sliders: User slider weights {"relevance":0.9,"followers":0.4,"user_interaction":0.7}.
    
    Returns:
        The same list of candidates, each with an added 'why_matched' field.
    """
    for item in candidates:
        meta = item.get("metadata", {}) or {}
        hashtags = " ".join([f"#{h}" for h in (meta.get("hashtags") or [])[:5]])
        author = meta.get("author") or meta.get("handle") or "Unknown"
        region = meta.get("region") or ""
        bio = meta.get("bio") or ""
        
        followers = meta.get("followers") or meta.get("follower_count")
        eng_rate = meta.get("engagement_rate") or meta.get("eng_rate")
        plays = meta.get("avg_plays") or meta.get("play_median")
        likes = meta.get("likes") or meta.get("avg_likes")
        comments = meta.get("comments") or meta.get("avg_comments")
        shares = meta.get("shares") or meta.get("avg_shares")

        user_prompt = f"""
Brand Query: {query}

Creator: {author} ({region})
Bio: {bio}
Hashtags: {hashtags}

Metrics:
- Followers: {followers}
- Engagement rate: {eng_rate}
- Avg plays: {plays}
- Likes: {likes}, Comments: {comments}, Shares: {shares}

User slider weights:
relevance={sliders.get('relevance', 0):.2f}, followers={sliders.get('followers', 0):.2f}, user_interaction={sliders.get('user_interaction', 0):.2f}

Task:
In one or two sentences, explain why this creator matches the brand query.
Mention topic/niche relevance and engagement evidence. Be concise, factual, and professional. No emojis.
"""

        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b" ,  
                messages=[
                    {"role": "system", "content": (
                        "You are an assistant that explains why a creator matches a brand query. "
                        "Focus on topic relevance, niche fit, and engagement quality."
                    )},
                    {"role": "user", "content": user_prompt.strip()},
                ],
                temperature=0.5,
                max_tokens=100
            )
            explanation = completion.choices[0].message.content.strip()
        except Exception as e:
            explanation = (
                f"Relevant creator in the target niche with strong engagement metrics and audience alignment. ({e})"
            )

        item["why_matched"] = explanation
    return candidates
