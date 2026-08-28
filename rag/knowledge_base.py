"""
RAG Knowledge Base Builder
FlyRank Content Intelligence Platform

Builds a ChromaDB vector store from:
  1. Built-in SEO guidelines (static)
  2. Live URL fetching — automatically scrapes given URLs for latest SEO knowledge
     and indexes them into the same vector store.

Usage:
    from rag.knowledge_base import build_knowledge_base
    retriever = build_knowledge_base()          # uses default static + default URLs
    retriever = build_knowledge_base(extra_urls=["https://example.com/seo-guide"])
"""
import os
import time
import requests
from bs4 import BeautifulSoup

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings")
os.makedirs(CHROMA_DIR, exist_ok=True)

COLLECTION_NAME = "flyrank_seo_knowledge"

# ── Static built-in SEO knowledge ─────────────────────────────────────────────
STATIC_DOCS = [
    {
        "id": "seo_titles",
        "text": (
            "SEO Title Optimization Rules:\n"
            "1. Keep titles between 50-60 characters.\n"
            "2. Place primary keyword near the beginning.\n"
            "3. Include the current year for time-sensitive content (e.g. '2026 Guide').\n"
            "4. Use power words: Best, Complete, Ultimate, Step-by-Step.\n"
            "5. Avoid keyword stuffing — write for humans first.\n"
            "6. Each page should have a unique title tag.\n"
            "7. Brand name at the end is optional: 'Title | Brand'."
        ),
        "metadata": {"category": "title_optimization", "source": "static"},
    },
    {
        "id": "meta_descriptions",
        "text": (
            "Meta Description Best Practices:\n"
            "1. Optimal length: 150-160 characters.\n"
            "2. Include primary and secondary keywords naturally.\n"
            "3. Use a clear call-to-action: 'Learn how', 'Discover', 'Find out'.\n"
            "4. Make it unique per page — duplicate meta descriptions hurt rankings.\n"
            "5. Summarize what the user will get from the page.\n"
            "6. Avoid generic phrases like 'Welcome to our website'."
        ),
        "metadata": {"category": "meta_description", "source": "static"},
    },
    {
        "id": "content_freshness",
        "text": (
            "Content Freshness & Refresh Guidelines:\n"
            "1. Content older than 365 days with declining CTR should be reviewed.\n"
            "2. Update statistics, dates, and examples to current year.\n"
            "3. Add new sections addressing recent developments in the topic.\n"
            "4. Internal linking: add links to newer related content.\n"
            "5. Content with >180 days since last update and position drop > 3 spots = high priority refresh.\n"
            "6. Evergreen content can be refreshed annually; trending content quarterly.\n"
            "7. After refresh, submit URL to Google Search Console for re-indexing."
        ),
        "metadata": {"category": "content_freshness", "source": "static"},
    },
    {
        "id": "ctr_optimization",
        "text": (
            "CTR Improvement Strategies:\n"
            "1. High impressions + low CTR (< 2%) = title/description mismatch.\n"
            "2. Test bracketed modifiers: [Guide], [2026], [Free], [Updated].\n"
            "3. Add numbers to titles: '7 Ways', 'Top 10', '5 Mistakes'.\n"
            "4. Schema markup (FAQ, HowTo, Review) increases rich snippet eligibility.\n"
            "5. Featured snippet optimization: answer questions directly in the first paragraph.\n"
            "6. Position 1-3 avg CTR: 28%, 15%, 11%. Below 2% at position 1-3 = urgent refresh."
        ),
        "metadata": {"category": "ctr_optimization", "source": "static"},
    },
    {
        "id": "keyword_strategy",
        "text": (
            "Keyword & Content Strategy:\n"
            "1. Target long-tail keywords for low-competition niches.\n"
            "2. Semantic SEO: cover related entities and LSI keywords in content.\n"
            "3. Keyword cannibalization: if 2+ pages target same keyword, consolidate.\n"
            "4. Search intent types: informational, navigational, transactional, commercial.\n"
            "5. Align content type to intent: blog posts for informational, landing pages for transactional.\n"
            "6. Use heading hierarchy (H1 > H2 > H3) to signal topic structure to search engines."
        ),
        "metadata": {"category": "keyword_strategy", "source": "static"},
    },
    {
        "id": "technical_seo",
        "text": (
            "Technical SEO Checklist:\n"
            "1. Core Web Vitals: LCP < 2.5s, FID < 100ms, CLS < 0.1.\n"
            "2. Mobile-first indexing: ensure pages are fully functional on mobile.\n"
            "3. Canonical tags: prevent duplicate content issues.\n"
            "4. Structured data: implement JSON-LD schema for articles, FAQs, products.\n"
            "5. Internal linking: every page should have at least 3 internal links.\n"
            "6. Image optimization: use alt tags, compress images, use WebP format.\n"
            "7. Page speed: compress JS/CSS, enable caching, use CDN."
        ),
        "metadata": {"category": "technical_seo", "source": "static"},
    },
    {
        "id": "content_scoring",
        "text": (
            "FlyRank Content Refresh Score Interpretation:\n"
            "Score 0.8 - 1.0: Immediate refresh required. High staleness + declining CTR.\n"
            "Score 0.6 - 0.79: Refresh recommended within 30 days.\n"
            "Score 0.4 - 0.59: Review content, minor updates may help.\n"
            "Score 0.0 - 0.39: Content performing well. Monitor only.\n"
            "Reason codes: DECAY = traffic decline, STALE = age-based, "
            "CTR_GAP = impressions high but clicks low, OPPORTUNITY = high volume low rank, "
            "REVIEW = manual assessment needed."
        ),
        "metadata": {"category": "scoring", "source": "static"},
    },
]

# Default URLs to auto-fetch for latest SEO knowledge
DEFAULT_SEO_URLS = [
    "https://developers.google.com/search/docs/fundamentals/seo-starter-guide",
    "https://moz.com/beginners-guide-to-seo",
]


# ── URL Fetching ──────────────────────────────────────────────────────────────
def fetch_url_content(url: str, max_chars: int = 3000, timeout: int = 10) -> str | None:
    """
    Fetches a URL, strips HTML tags, returns plain text (truncated).
    Returns None on failure — never crashes the pipeline.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (FlyRank Content Intelligence Bot)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove nav, footer, scripts, styles
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse whitespace
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)[:max_chars]
        print(f"  [knowledge_base] ✓ Fetched {len(text)} chars from {url}")
        return text
    except Exception as e:
        print(f"  [knowledge_base] ✗ Could not fetch {url}: {e}")
        return None


def fetch_url_docs(urls: list[str]) -> list[dict]:
    """Fetch multiple URLs and return as document dicts."""
    docs = []
    for i, url in enumerate(urls):
        content = fetch_url_content(url)
        if content:
            docs.append({
                "id":       f"web_{i}_{int(time.time())}",
                "text":     f"[Source: {url}]\n\n{content}",
                "metadata": {"category": "web_knowledge", "source": url},
            })
        time.sleep(0.5)  # polite delay
    return docs


# ── ChromaDB Setup ────────────────────────────────────────────────────────────
def build_knowledge_base(extra_urls: list[str] | None = None, force_rebuild: bool = False):
    """
    Builds (or loads existing) ChromaDB knowledge base.
    
    Args:
        extra_urls: Additional URLs to scrape and index beyond defaults.
        force_rebuild: If True, clears and rebuilds the collection.
    
    Returns:
        ChromaDB collection object.
    """
    try:
        import chromadb
    except ImportError:
        print("[knowledge_base] chromadb not installed. Returning None.")
        return None

    try:
        from langchain_openai import OpenAIEmbeddings
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from Agents.model import API_KEY, BASE_URL
        embed_fn = OpenAIEmbeddings(
            openai_api_key=API_KEY or "placeholder",
            openai_api_base=BASE_URL if BASE_URL else None,
        ) if API_KEY else None
    except Exception:
        embed_fn = None

    # Use ChromaDB with simple embedding (no OpenAI needed for basic setup)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Check if already populated
    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        print(f"[knowledge_base] Loaded existing collection ({existing_count} docs). Skipping rebuild.")
        return collection

    # Collect all documents
    all_docs = list(STATIC_DOCS)

    # Auto-fetch default + extra URLs
    urls_to_fetch = list(DEFAULT_SEO_URLS) + (extra_urls or [])
    print(f"\n[knowledge_base] Fetching {len(urls_to_fetch)} URLs for latest SEO knowledge...")
    url_docs = fetch_url_docs(urls_to_fetch)
    all_docs.extend(url_docs)

    # Add to ChromaDB
    ids      = [d["id"]   for d in all_docs]
    texts    = [d["text"] for d in all_docs]
    metas    = [d["metadata"] for d in all_docs]

    # ChromaDB uses its own embedding by default (all-MiniLM-L6-v2 via sentence-transformers)
    # If OpenAI embeddings are available, we pass them — otherwise ChromaDB handles it
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metas,
    )

    print(f"[knowledge_base] ✓ Knowledge base built: {collection.count()} documents")
    return collection


if __name__ == "__main__":
    col = build_knowledge_base()
    if col:
        results = col.query(query_texts=["how to improve CTR"], n_results=2)
        print("\n=== Test Query: 'how to improve CTR' ===")
        for doc in results["documents"][0]:
            print(doc[:200], "\n---")
