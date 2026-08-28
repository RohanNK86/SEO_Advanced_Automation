"""
Feature Engineering Pipeline
FlyRank Content Intelligence Platform
Derives advanced SEO features from raw dataset and saves processed output.
"""
import os
import pandas as pd
import numpy as np

RAW_CSV = os.path.join(os.path.dirname(__file__), "..", "Datasets", "content_refresh_anonymized (1).csv")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_CSV  = os.path.join(OUT_DIR, "features.csv")


def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[engineer] Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "unknown")
    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    # CTR delta: impressions grew but CTR fell → opportunity
    safe_impr = df["impressions_prev_30d"].replace(0, np.nan)
    df["impression_growth"] = (df["impressions_last_30d"] - df["impressions_prev_30d"]) / safe_impr
    df["impression_growth"] = df["impression_growth"].fillna(0).clip(-5, 5)

    safe_clicks_prev = df["clicks_prev_30d"].replace(0, np.nan)
    df["click_growth"] = (df["clicks_last_30d"] - df["clicks_prev_30d"]) / safe_clicks_prev
    df["click_growth"] = df["click_growth"].fillna(0).clip(-5, 5)

    # CTR gap: expected CTR at position vs actual
    # Rough benchmark: pos 1 → 28%, pos 3 → 11%, pos 5 → 7%, pos 10 → 2%
    df["expected_ctr"] = 0.28 / (df["avg_position"].clip(lower=1) ** 0.7)
    df["ctr_delta"] = df["expected_ctr"] - df["ctr"]          # positive = underperforming
    df["ctr_delta"] = df["ctr_delta"].clip(-1, 1)

    # Freshness score: higher = stale
    max_age = df["days_since_last_update"].max()
    df["freshness_score"] = df["days_since_last_update"] / max_age

    # Decay score: traffic fell & position dropped
    df["decay_score"] = (
        (df["click_growth"] < -0.1).astype(int) +
        (df["impression_growth"] < -0.1).astype(int)
    ) / 2.0

    # Opportunity score: high impressions + low CTR + good search volume
    max_sv = df["search_volume"].replace(0, np.nan).max()
    df["opportunity_score"] = (
        (df["impressions_last_30d"] / (df["impressions_last_30d"].max() + 1)) * 0.4 +
        df["ctr_delta"].clip(0, 1) * 0.4 +
        (df["search_volume"] / (max_sv + 1)).fillna(0) * 0.2
    )

    # Engagement quality
    safe_sess = df["sessions_90d"].replace(0, np.nan)
    df["eng_quality"] = (df["engaged_sessions_90d"] / safe_sess).fillna(0)

    print(f"[engineer] Derived 8 new features. Total cols: {df.shape[1]}")
    return df


def run(path: str = RAW_CSV) -> pd.DataFrame:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_raw(path)
    df = impute(df)
    df = derive_features(df)
    df.to_csv(OUT_CSV, index=False)
    print(f"[engineer] Saved → {OUT_CSV}")
    return df


if __name__ == "__main__":
    run()
