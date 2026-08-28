"""
ML Prediction Service
FlyRank Content Intelligence Platform
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Agents.seo_agent import get_agent, score_to_priority
from backend.schemas import PageMetrics, PredictResponse


def predict(page: PageMetrics) -> PredictResponse:
    """Run ML model prediction for a single page."""
    agent = get_agent()
    page_dict = page.model_dump()

    # ML-only analysis (no RAG)
    result = agent.analyze(page_dict, use_rag=False)

    return PredictResponse(
        content_id=    page.content_id,
        ml_score=      result["ml_score"],
        priority=      result["priority"],
        priority_label=result["priority_label"],
        priority_color=result["priority_color"],
        reason_codes=  result["reason_codes"],
        reason_labels= result["reason_labels"],
        confidence=    result["confidence"],
        ml_action=     result["ml_action"],
    )
