"""
Pydantic Schemas — Request & Response Models
FlyRank Content Intelligence Platform
"""
from pydantic import BaseModel, Field
from typing import Optional, List


# ── Request Models ────────────────────────────────────────────────────────────

class PageMetrics(BaseModel):
    """Raw page metrics for ML prediction / agent analysis."""
    content_id:            Optional[str]   = None
    url:                   Optional[str]   = None
    ctr:                   float           = Field(0.0,   description="Click-through rate (0-1)")
    impressions_90d:       float           = Field(0.0)
    clicks_90d:            float           = Field(0.0)
    sessions_90d:          float           = Field(0.0)
    avg_position:          float           = Field(10.0)
    content_age_days:      float           = Field(0.0)
    days_since_last_update:float           = Field(0.0)
    trend_pct:             float           = Field(0.0)
    search_volume:         float           = Field(0.0)
    competition:           float           = Field(0.0)
    cpc:                   float           = Field(0.0)
    word_count:            float           = Field(0.0)
    staleness_score:       float           = Field(0.0)
    ctr_gap_score:         float           = Field(0.0)
    volume_score:          float           = Field(0.0)
    engagement_rate:       float           = Field(0.0)
    scroll_rate:           float           = Field(0.0)
    impressions_last_30d:  float           = Field(0.0)
    clicks_last_30d:       float           = Field(0.0)
    impressions_prev_30d:  float           = Field(0.0)
    clicks_prev_30d:       float           = Field(0.0)
    # Existing score/action from dataset (optional)
    score:                 Optional[float] = None
    action:                Optional[str]   = None
    reason_code:           Optional[str]   = None


class AnalyzeRequest(BaseModel):
    page: PageMetrics
    use_rag:    bool        = True
    extra_urls: List[str]   = Field(default_factory=list,
        description="Live URLs to fetch and add to RAG knowledge base")


class ApproveRequest(BaseModel):
    content_id: str
    decision:   str   = Field(..., description="APPROVE | REJECT | EDIT")
    notes:      Optional[str] = None


class AddKnowledgeURLRequest(BaseModel):
    url: str
    description: Optional[str] = None


# ── Response Models ───────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    content_id:     Optional[str]
    ml_score:       float
    priority:       str
    priority_label: str
    priority_color: str
    reason_codes:   List[str]
    reason_labels:  List[str]
    confidence:     float
    ml_action:      str


class AnalyzeResponse(BaseModel):
    content_id:          Optional[str]
    ml_score:            float
    priority:            str
    priority_label:      str
    priority_color:      str
    reason_codes:        List[str]
    reason_labels:       List[str]
    confidence:          float
    ml_action:           str
    rag_docs_count:      int
    llm_available:       bool
    llm_recommendation:  Optional[str]
    context_snippet:     Optional[str]


class PageSummary(BaseModel):
    content_id:     str
    client_id:      Optional[str]
    score:          float
    action:         str
    reason_code:    Optional[str]
    ctr:            float
    avg_position:   float
    impressions_90d:float
    content_age_days:float
    trend_pct:      float
    priority:       str
    priority_color: str


class DashboardMetrics(BaseModel):
    total_pages:       int
    needs_refresh:     int
    review_queue:      int
    healthy_pages:     int
    avg_score:         float
    avg_ctr:           float
    avg_position:      float
    critical_count:    int
    high_count:        int
    medium_count:      int
    low_count:         int
