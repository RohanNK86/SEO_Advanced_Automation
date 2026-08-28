"""
Leakage Audit + Time-Based Split
FlyRank Content Intelligence Platform

Checks for data leakage (future metrics in training set) and
enforces time-based train/test split using content_age_days.
"""
import os
import pandas as pd
import numpy as np

SUSPICIOUS_FEATURES = [
    # These are computed from clicks/impressions — may encode future info if not careful
    "score",           # target — never use as input feature
    "reason_code",     # derived label — not a raw feature
    "action",          # derived label — not a raw feature
    "ctr_gap_score",   # derived from future CTR knowledge
    "volume_score",    # derived score — verify it's pre-refresh
]

ALLOWED_INPUT_FEATURES = [
    "search_volume", "competition", "cpc",
    "word_count", "char_count",
    "impressions_90d", "clicks_90d", "pageviews_90d", "sessions_90d",
    "engaged_sessions_90d", "scroll_events_90d",
    "days_with_impressions", "days_with_sessions",
    "impressions_last_30d", "clicks_last_30d", "sessions_last_30d",
    "impressions_prev_30d", "clicks_prev_30d", "sessions_prev_30d",
    "content_age_days", "days_since_last_update",
    "ctr", "avg_position", "engagement_rate", "scroll_rate", "ai_traffic_pct",
    "trend_pct", "staleness_score",
    # Derived in engineer.py (safe — all look-back only)
    "impression_growth", "click_growth", "ctr_delta",
    "freshness_score", "decay_score", "opportunity_score", "eng_quality",
]

TARGET_COL  = "score"
ACTION_COL  = "action"
SPLIT_COL   = "content_age_days"
SPLIT_PCTILE = 80  # train on older 80% of content, test on newest 20%


def audit(df: pd.DataFrame) -> dict:
    results = {
        "flagged_in_features": [],
        "duplicate_rows": int(df.duplicated().sum()),
        "null_target_rows": int(df[TARGET_COL].isna().sum()),
    }

    # Check if suspicious features appear in dataset
    for f in SUSPICIOUS_FEATURES:
        if f in df.columns:
            results["flagged_in_features"].append(f)

    # Correlation of potential leakers with target
    leakage_corr = {}
    for f in SUSPICIOUS_FEATURES:
        if f in df.columns and f != TARGET_COL:
            try:
                corr = df[f].corr(df[TARGET_COL])
                leakage_corr[f] = round(float(corr), 4)
            except Exception:
                pass
    results["leakage_correlations"] = leakage_corr

    return results


def time_based_split(df: pd.DataFrame, split_col: str = SPLIT_COL, pctile: int = SPLIT_PCTILE):
    """
    Split by content age: older content → train, newer content → test.
    This prevents future-data leakage from newer-to-older direction.
    """
    threshold = np.percentile(df[split_col].dropna(), pctile)
    train = df[df[split_col] >= threshold].copy()   # older = higher age_days
    test  = df[df[split_col] < threshold].copy()    # newer = lower age_days
    print(f"[leakage_audit] Time split at age_days={threshold:.0f}")
    print(f"  Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    return train, test


def run(df: pd.DataFrame):
    print("\n=== LEAKAGE AUDIT ===")
    results = audit(df)
    print(f"  Duplicate rows    : {results['duplicate_rows']}")
    print(f"  Null target rows  : {results['null_target_rows']}")
    print(f"  Flagged features  : {results['flagged_in_features']}")
    print(f"  Leakage corr      : {results['leakage_correlations']}")

    # Safe feature set
    safe_features = [f for f in ALLOWED_INPUT_FEATURES if f in df.columns]
    print(f"\n  Safe input features ({len(safe_features)}): {safe_features}")

    # Time-based split
    train, test = time_based_split(df)
    return train, test, safe_features, results


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from feature_pipeline.engineer import run as engineer_run
    df = engineer_run()
    run(df)
