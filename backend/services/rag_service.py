"""
RAG Service — Knowledge Base Query + Live URL Fetching
FlyRank Content Intelligence Platform
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def query(question: str) -> str:
    """Query the SEO knowledge base with a natural language question."""
    from Agents.RAG import query_knowledge
    return query_knowledge(question)


def add_url(url: str) -> bool:
    """Add a live URL to the RAG knowledge base at runtime."""
    from rag.retriever import add_url_to_knowledge
    return add_url_to_knowledge(url)


def get_knowledge_stats() -> dict:
    """Return stats about the current knowledge base."""
    try:
        from rag.retriever import get_collection
        col = get_collection()
        if col:
            return {"total_documents": col.count(), "status": "active"}
    except Exception as e:
        return {"total_documents": 0, "status": f"error: {e}"}
    return {"total_documents": 0, "status": "not_initialized"}
