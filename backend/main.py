"""
FastAPI Main Application
FlyRank Content Intelligence Platform

Run with:
    uvicorn backend.main:app --reload --port 8000

Docs at: http://localhost:8000/docs
"""
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import pandas as pd
import numpy as np

from backend.schemas import (
    PageMetrics, AnalyzeRequest, ApproveRequest, AddKnowledgeURLRequest,
    PredictResponse, AnalyzeResponse, PageSummary, DashboardMetrics,
)
from backend.services import predict_service, rag_service, agent_service
from Agents.seo_agent import score_to_priority

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FlyRank Content Intelligence Platform",
    description=(
        "End-to-End ML + RAG + AI Agent platform for SEO content optimization. "
        "Predicts refresh priority, generates AI recommendations, and tracks content performance."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Dataset cache ─────────────────────────────────────────────────────────────
_df: Optional[pd.DataFrame] = None
_approvals: dict = {}  # content_id → decision

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv")
RAW_DATASET  = os.path.join(os.path.dirname(__file__), "..", "Datasets", "content_refresh_anonymized (1).csv")


def get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        path = DATASET_PATH if os.path.exists(DATASET_PATH) else RAW_DATASET
        _df = pd.read_csv(path)
        # Ensure numeric score
        _df["score"] = pd.to_numeric(_df["score"], errors="coerce").fillna(0.5)
        # Add priority info
        _df["priority"] = _df["score"].apply(lambda s: score_to_priority(s)["priority"])
        _df["priority_color"] = _df["score"].apply(lambda s: score_to_priority(s)["color"])
        print(f"[main] Dataset loaded: {len(_df):,} rows")
    return _df


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "dataset_rows": len(get_df()),
        "knowledge_base": rag_service.get_knowledge_stats(),
    }


# ── Dashboard ──────────────────────────────────────────────────────────────────
@app.get("/api/dashboard", response_model=DashboardMetrics, tags=["Analytics"])
def dashboard():
    df = get_df()
    total = len(df)
    score = df["score"].fillna(0)

    needs_refresh = int((score >= 0.6).sum())
    review_queue  = int(((score >= 0.4) & (score < 0.6)).sum())
    healthy       = int((score < 0.4).sum())
    critical      = int((score >= 0.8).sum())
    high          = int(((score >= 0.6) & (score < 0.8)).sum())
    medium        = int(((score >= 0.4) & (score < 0.6)).sum())
    low           = int((score < 0.4).sum())

    return DashboardMetrics(
        total_pages=    total,
        needs_refresh=  needs_refresh,
        review_queue=   review_queue,
        healthy_pages=  healthy,
        avg_score=      round(float(score.mean()), 4),
        avg_ctr=        round(float(df["ctr"].fillna(0).mean()), 4),
        avg_position=   round(float(df["avg_position"].fillna(10).mean()), 2),
        critical_count= critical,
        high_count=     high,
        medium_count=   medium,
        low_count=      low,
    )


# ── Pages List ─────────────────────────────────────────────────────────────────
@app.get("/api/pages", response_model=List[PageSummary], tags=["Pages"])
def list_pages(
    page:     int   = Query(1,  ge=1,   description="Page number"),
    per_page: int   = Query(20, ge=1, le=100, description="Items per page"),
    sort_by:  str   = Query("score", description="Column to sort by"),
    order:    str   = Query("desc", description="asc or desc"),
    priority: Optional[str] = Query(None, description="Filter: CRITICAL|HIGH|MEDIUM|LOW"),
    search:   Optional[str] = Query(None, description="Search by content_id or client_id"),
):
    df = get_df().copy()

    # Filter
    if priority:
        df = df[df["priority"] == priority.upper()]
    if search:
        mask = (
            df["content_id"].astype(str).str.contains(search, case=False, na=False) |
            df["client_id"].astype(str).str.contains(search, case=False, na=False)
        )
        df = df[mask]

    # Sort
    asc = order.lower() != "desc"
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=asc, na_position="last")

    # Paginate
    start = (page - 1) * per_page
    end   = start + per_page
    slice_df = df.iloc[start:end]

    results = []
    for _, row in slice_df.iterrows():
        results.append(PageSummary(
            content_id=     str(row.get("content_id", "")),
            client_id=      str(row.get("client_id", "")),
            score=          float(row.get("score", 0)),
            action=         str(row.get("action", "Review")),
            reason_code=    str(row.get("reason_code", "")),
            ctr=            float(row.get("ctr", 0)),
            avg_position=   float(row.get("avg_position", 10)),
            impressions_90d=float(row.get("impressions_90d", 0)),
            content_age_days=float(row.get("content_age_days", 0)),
            trend_pct=      float(row.get("trend_pct", 0)),
            priority=       str(row.get("priority", "MEDIUM")),
            priority_color= str(row.get("priority_color", "#eab308")),
        ))
    return results


# ── Page Detail ───────────────────────────────────────────────────────────────
@app.get("/api/pages/{content_id}", tags=["Pages"])
def get_page(content_id: str):
    df = get_df()
    row = df[df["content_id"] == content_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Page '{content_id}' not found")
    data = row.iloc[0].where(pd.notnull(row.iloc[0]), other=None).to_dict()
    # Attach approval status
    data["approval_status"] = _approvals.get(content_id, "PENDING")
    return data


# ── ML Predict ────────────────────────────────────────────────────────────────
@app.post("/api/predict", response_model=PredictResponse, tags=["ML"])
def predict(page: PageMetrics):
    """Fast ML-only refresh score prediction (no LLM)."""
    try:
        return predict_service.predict(page)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Full Agent Analysis ───────────────────────────────────────────────────────
@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["AI Agent"])
def analyze(request: AnalyzeRequest):
    """
    Full AI analysis: ML prediction + RAG knowledge retrieval + LLM recommendation.
    Pass extra_urls to automatically fetch and index live SEO articles.
    """
    try:
        return agent_service.full_analyze(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Analyze page from dataset by ID ──────────────────────────────────────────
@app.get("/api/analyze/{content_id}", response_model=AnalyzeResponse, tags=["AI Agent"])
def analyze_by_id(content_id: str, use_rag: bool = True):
    """Analyze a page from the dataset by content_id."""
    df = get_df()
    row = df[df["content_id"] == content_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Page '{content_id}' not found")

    page_dict = row.iloc[0].where(pd.notnull(row.iloc[0]), other=0).to_dict()
    page = PageMetrics(**{k: page_dict.get(k) for k in PageMetrics.model_fields if k in page_dict})
    page.content_id = content_id

    req = AnalyzeRequest(page=page, use_rag=use_rag)
    try:
        return agent_service.full_analyze(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Approve / Reject ──────────────────────────────────────────────────────────
@app.post("/api/approve", tags=["Workflow"])
def approve(request: ApproveRequest):
    """Record human approval/rejection for a page refresh."""
    if request.decision.upper() not in ("APPROVE", "REJECT", "EDIT"):
        raise HTTPException(status_code=400, detail="decision must be APPROVE, REJECT, or EDIT")
    _approvals[request.content_id] = request.decision.upper()
    return {
        "content_id": request.content_id,
        "decision":   request.decision.upper(),
        "notes":      request.notes,
        "status":     "recorded",
    }


# ── RAG Knowledge Base ────────────────────────────────────────────────────────
@app.post("/api/knowledge/add-url", tags=["RAG"])
def add_knowledge_url(request: AddKnowledgeURLRequest):
    """Add a live URL to the RAG knowledge base. Content is fetched and indexed automatically."""
    success = rag_service.add_url(request.url)
    if not success:
        raise HTTPException(status_code=422, detail=f"Could not fetch or index URL: {request.url}")
    return {"url": request.url, "status": "indexed", "knowledge_stats": rag_service.get_knowledge_stats()}


@app.get("/api/knowledge/query", tags=["RAG"])
def query_knowledge(q: str = Query(..., description="SEO question to ask the knowledge base")):
    """Query the SEO knowledge base directly."""
    answer = rag_service.query(q)
    return {"question": q, "answer": answer}


@app.get("/api/knowledge/stats", tags=["RAG"])
def knowledge_stats():
    return rag_service.get_knowledge_stats()


# ── Metrics Summary ───────────────────────────────────────────────────────────
@app.get("/api/metrics", tags=["Analytics"])
def metrics():
    df = get_df()
    score = df["score"].fillna(0)
    return {
        "action_distribution": df["action"].value_counts().to_dict(),
        "priority_distribution": df["priority"].value_counts().to_dict(),
        "score_percentiles": {
            "p25": round(float(score.quantile(0.25)), 4),
            "p50": round(float(score.quantile(0.50)), 4),
            "p75": round(float(score.quantile(0.75)), 4),
            "p90": round(float(score.quantile(0.90)), 4),
        },
        "ctr_stats": {
            "mean": round(float(df["ctr"].fillna(0).mean()), 4),
            "median": round(float(df["ctr"].fillna(0).median()), 4),
        },
        "total_approvals": len(_approvals),
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
