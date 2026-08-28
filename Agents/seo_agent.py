"""
SEO Analysis Agent
FlyRank Content Intelligence Platform

A single unified agent that combines:
1. ML score prediction (from saved model)
2. RAG context retrieval (ChromaDB + live URL fetching)
3. LLM recommendation generation (when API key is set)

Returns a structured analysis result used by the FastAPI backend.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Reason code → human readable labels
REASON_LABELS = {
    "DECAY":       "Traffic is declining month-over-month",
    "STALE":       "Content has not been updated in over a year",
    "CTR_GAP":     "High impressions but very low click-through rate",
    "OPPORTUNITY": "High search volume but poor ranking position",
    "REVIEW":      "Multiple signals suggest manual review needed",
}

# Score → action thresholds
def score_to_priority(score: float) -> dict:
    if score >= 0.8:
        return {"priority": "CRITICAL",  "label": "Refresh Immediately", "color": "#ef4444"}
    elif score >= 0.6:
        return {"priority": "HIGH",      "label": "Refresh Soon",        "color": "#f97316"}
    elif score >= 0.4:
        return {"priority": "MEDIUM",    "label": "Review Content",       "color": "#eab308"}
    else:
        return {"priority": "LOW",       "label": "Monitor Only",         "color": "#22c55e"}


class SEOAnalysisAgent:
    """
    Unified SEO analysis agent.
    Usage:
        agent = SEOAnalysisAgent()
        result = agent.analyze(page_data_dict)
    """

    def __init__(self):
        self._model       = None
        self._feat_cols   = None
        self._classifier  = None
        self._clf_scaler  = None
        self._clf_encoder = None
        self._clf_feats   = None
        self._load_models()

    def _load_models(self):
        try:
            self._model     = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
            self._feat_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
            print("[SEOAgent] ✓ Regression model loaded")
        except FileNotFoundError:
            print("[SEOAgent] ⚠ No trained model found. Run MLModels/train_pipeline.py first.")

        try:
            self._classifier  = joblib.load(os.path.join(MODELS_DIR, "action_classifier.pkl"))
            self._clf_scaler  = joblib.load(os.path.join(MODELS_DIR, "action_scaler.pkl"))
            self._clf_encoder = joblib.load(os.path.join(MODELS_DIR, "action_label_encoder.pkl"))
            self._clf_feats   = joblib.load(os.path.join(MODELS_DIR, "classifier_features.pkl"))
            print("[SEOAgent] ✓ Classifier model loaded")
        except FileNotFoundError:
            print("[SEOAgent] ⚠ No classifier found. Run MLModels/TreeClassify.py first.")

    def _predict_score(self, page_data: dict) -> float | None:
        """Run regression model to predict refresh score."""
        if self._model is None or self._feat_cols is None:
            return None
        import pandas as pd
        row = {c: [page_data.get(c, 0)] for c in self._feat_cols}
        df  = pd.DataFrame(row).fillna(0)
        score = float(self._model.predict(df)[0])
        return round(min(max(score, 0.0), 1.0), 4)

    def _predict_action(self, page_data: dict) -> str | None:
        """Run classifier to predict action label."""
        if self._classifier is None:
            return None
        import pandas as pd
        row = {c: [page_data.get(c, 0)] for c in self._clf_feats}
        df  = pd.DataFrame(row).fillna(0)
        for c in df.select_dtypes(include=["object"]).columns:
            df[c] = 0
        scaled  = self._clf_scaler.transform(df)
        label_i = self._classifier.predict(scaled)[0]
        return self._clf_encoder.inverse_transform([label_i])[0]

    def _derive_reason_codes(self, page_data: dict) -> list[str]:
        """Derive reason codes from raw page metrics."""
        reasons = []
        ctr  = page_data.get("ctr", 0)
        impr = page_data.get("impressions_90d", 0)
        age  = page_data.get("content_age_days", 0)
        pos  = page_data.get("avg_position", 10)
        trend= page_data.get("trend_pct", 0)
        sv   = page_data.get("search_volume", 0)

        if trend < -15:
            reasons.append("DECAY")
        if age > 365:
            reasons.append("STALE")
        if ctr < 0.02 and impr > 100:
            reasons.append("CTR_GAP")
        if sv > 500 and pos > 6:
            reasons.append("OPPORTUNITY")
        if not reasons:
            reasons.append("REVIEW")

        return reasons

    def analyze(self, page_data: dict, use_rag: bool = True, extra_urls: list[str] | None = None) -> dict:
        """
        Full page analysis.
        
        Args:
            page_data:  Dict of page metrics (from dataset row or API input).
            use_rag:    Whether to run RAG retrieval + LLM recommendation.
            extra_urls: Optional live URLs to fetch and incorporate into RAG context.
        
        Returns:
            Structured analysis result dict.
        """
        # 1. ML prediction
        ml_score  = self._predict_score(page_data)
        ml_action = self._predict_action(page_data)
        reason_codes = self._derive_reason_codes(page_data)

        # Use ML score or fall back to dataset score
        final_score = ml_score if ml_score is not None else page_data.get("score", 0.5)
        priority_info = score_to_priority(final_score)

        # 2. RAG + LLM (optional)
        rag_result = {}
        if use_rag:
            try:
                # If extra_urls provided, add them to knowledge base first
                if extra_urls:
                    from rag.retriever import add_url_to_knowledge
                    for url in extra_urls:
                        add_url_to_knowledge(url)

                from Agents.RAG import analyze_page_with_rag
                rag_result = analyze_page_with_rag(page_data)
            except Exception as e:
                rag_result = {"error": str(e)}

        return {
            # ML outputs
            "ml_score":      final_score,
            "ml_action":     ml_action or page_data.get("action", "Review"),
            "reason_codes":  reason_codes,
            "reason_labels": [REASON_LABELS.get(r, r) for r in reason_codes],
            "priority":      priority_info["priority"],
            "priority_label":priority_info["label"],
            "priority_color":priority_info["color"],
            "confidence":    round(min(final_score * 1.2, 1.0) * 100, 1),

            # RAG outputs
            "rag_docs_count":    len(rag_result.get("retrieved_docs", [])),
            "llm_recommendation":rag_result.get("llm_recommendation"),
            "llm_available":     rag_result.get("llm_available", False),
            "context_snippet":   rag_result.get("context_used", "")[:300],
        }


# Singleton instance for backend reuse
_agent_instance = None

def get_agent() -> SEOAnalysisAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = SEOAnalysisAgent()
    return _agent_instance


if __name__ == "__main__":
    agent = SEOAnalysisAgent()
    sample = {
        "ctr": 0.012,
        "impressions_90d": 650,
        "avg_position": 7.1,
        "content_age_days": 430,
        "days_since_last_update": 400,
        "trend_pct": -25.0,
        "search_volume": 1200,
        "score": 0.75,
        "action": "Refresh Content",
    }
    result = agent.analyze(sample, use_rag=True)
    print("\n=== SEO Agent Analysis ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
