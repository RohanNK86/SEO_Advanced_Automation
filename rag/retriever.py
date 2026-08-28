"""
ChromaDB Retriever Wrapper
FlyRank Content Intelligence Platform

Wraps the ChromaDB collection with a simple query interface
used by both the agent and the backend services.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_collection = None


def get_collection():
    """Lazy-load or return existing ChromaDB collection."""
    global _collection
    if _collection is None:
        from rag.knowledge_base import build_knowledge_base
        _collection = build_knowledge_base()
    return _collection


def retrieve(query: str, n_results: int = 3, category: str | None = None) -> list[dict]:
    """
    Retrieve relevant SEO knowledge for a query.

    Args:
        query:     The natural-language query (e.g. "low CTR high impressions").
        n_results: Number of documents to return.
        category:  Optional filter by metadata category.

    Returns:
        List of dicts with keys: text, category, source, distance.
    """
    collection = get_collection()
    if collection is None:
        return []

    where = {"category": category} if category else None
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            where=where,
        )
    except Exception as e:
        print(f"[retriever] Query error: {e}")
        return []

    docs = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({
            "text":     text,
            "category": meta.get("category", "unknown"),
            "source":   meta.get("source", "static"),
            "distance": round(float(dist), 4),
        })
    return docs


def retrieve_for_page(page_data: dict, n_results: int = 4) -> list[dict]:
    """
    Build an automatic query from page metrics and retrieve relevant knowledge.
    
    This is the auto-URL-to-knowledge flow: given page data, it constructs
    a semantic query and fetches matching SEO guidelines.
    """
    parts = []
    ctr = page_data.get("ctr", 0)
    impr = page_data.get("impressions_90d", 0)
    pos = page_data.get("avg_position", 10)
    age = page_data.get("content_age_days", 0)
    score = page_data.get("score", 0)
    reason = page_data.get("reason_code", "")

    if ctr < 0.02 and impr > 100:
        parts.append("high impressions low CTR optimization")
    if pos > 5:
        parts.append("improve search ranking position")
    if age > 365:
        parts.append("content freshness refresh old content")
    if score > 0.6:
        parts.append("urgent content refresh needed declining traffic")
    if reason:
        parts.append(f"content issue {reason}")

    query = " ".join(parts) or "SEO content optimization best practices"
    return retrieve(query, n_results=n_results)


def add_url_to_knowledge(url: str) -> bool:
    """
    Dynamically add a new URL to the knowledge base at runtime.
    The URL is fetched, parsed, and added to ChromaDB immediately.
    """
    from rag.knowledge_base import fetch_url_content
    import time

    collection = get_collection()
    if collection is None:
        return False

    content = fetch_url_content(url)
    if not content:
        return False

    doc_id = f"runtime_{int(time.time())}"
    collection.add(
        ids=[doc_id],
        documents=[f"[Source: {url}]\n\n{content}"],
        metadatas=[{"category": "web_knowledge", "source": url}],
    )
    print(f"[retriever] Added URL to knowledge base: {url} (id={doc_id})")
    return True


if __name__ == "__main__":
    sample_page = {
        "ctr": 0.008,
        "impressions_90d": 500,
        "avg_position": 7.2,
        "content_age_days": 400,
        "score": 0.75,
        "reason_code": "DECAY",
    }
    docs = retrieve_for_page(sample_page)
    print(f"\nRetrieved {len(docs)} docs for sample page:")
    for d in docs:
        print(f"  [{d['category']}] dist={d['distance']} — {d['text'][:120]}...")
