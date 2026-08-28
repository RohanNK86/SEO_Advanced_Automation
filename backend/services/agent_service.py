"""
Agent Orchestration Service
FlyRank Content Intelligence Platform
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Agents.seo_agent import get_agent
from backend.schemas import PageMetrics, AnalyzeRequest, AnalyzeResponse


def full_analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run full SEO agent analysis: ML + RAG + LLM."""
    agent = get_agent()
    page_dict = request.page.model_dump()

    result = agent.analyze(
        page_data=page_dict,
        use_rag=request.use_rag,
        extra_urls=request.extra_urls if request.extra_urls else None,
    )

    return AnalyzeResponse(
        content_id=         request.page.content_id,
        ml_score=           result["ml_score"],
        priority=           result["priority"],
        priority_label=     result["priority_label"],
        priority_color=     result["priority_color"],
        reason_codes=       result["reason_codes"],
        reason_labels=      result["reason_labels"],
        confidence=         result["confidence"],
        ml_action=          result["ml_action"],
        rag_docs_count=     result["rag_docs_count"],
        llm_available=      result["llm_available"],
        llm_recommendation= result.get("llm_recommendation"),
        context_snippet=    result.get("context_snippet"),
    )
