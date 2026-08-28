"""
Full ML Training Pipeline
FlyRank Content Intelligence Platform

Trains Random Forest + XGBoost for refresh score prediction.
Logs all experiments to MLflow. Saves best model.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[train] xgboost not installed — skipping XGBoost experiment")

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("[train] mlflow not installed — skipping experiment tracking")

from feature_pipeline.engineer import run as engineer_run
from feature_pipeline.leakage_audit import run as audit_run, ALLOWED_INPUT_FEATURES, TARGET_COL

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

EXPERIMENT_NAME = "FlyRank-ContentRefresh-Score"


def encode_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Label-encode any remaining categoricals."""
    df = df[feature_cols].copy()
    for c in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
    return df


def evaluate(y_true, y_pred, label: str) -> dict:
    metrics = {
        "mae":  round(mean_absolute_error(y_true, y_pred), 6),
        "mse":  round(mean_squared_error(y_true, y_pred), 6),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 6),
        "r2":   round(r2_score(y_true, y_pred), 4),
    }
    print(f"\n  [{label}] MAE={metrics['mae']}  RMSE={metrics['rmse']}  R²={metrics['r2']}")
    return metrics


def train_model(name: str, model, X_train, y_train, X_test, y_test, params: dict):
    """Train, evaluate, log to MLflow, return metrics."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds, name)

    if HAS_MLFLOW:
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")
            if hasattr(model, "feature_importances_"):
                fi = pd.Series(model.feature_importances_, index=X_train.columns)
                top5 = fi.nlargest(5).to_dict()
                mlflow.log_dict(top5, "feature_importance_top5.json")

    return model, metrics


def run():
    print("=" * 55)
    print("  FlyRank ML Training Pipeline")
    print("=" * 55)

    # 1. Feature engineering
    df = engineer_run()

    # 2. Leakage audit + time-based split
    train_df, test_df, safe_features, _ = audit_run(df)

    # 3. Prepare X/y
    X_train = encode_features(train_df, safe_features)
    X_test  = encode_features(test_df,  safe_features)
    y_train = train_df[TARGET_COL].fillna(train_df[TARGET_COL].median())
    y_test  = test_df[TARGET_COL].fillna(test_df[TARGET_COL].median())

    print(f"\n[train] X_train={X_train.shape}  X_test={X_test.shape}")

    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 4. Setup MLflow
    if HAS_MLFLOW:
        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"[train] MLflow experiment: {EXPERIMENT_NAME}")

    results = {}

    # 5. Random Forest
    rf_params = {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5, "random_state": 42}
    rf = RandomForestRegressor(**rf_params, n_jobs=-1)
    rf_model, rf_metrics = train_model("RandomForest", rf, X_train, y_train, X_test, y_test, rf_params)
    results["RandomForest"] = (rf_model, rf_metrics)

    # 6. Gradient Boosting
    gb_params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5, "random_state": 42}
    gb = GradientBoostingRegressor(**gb_params)
    gb_model, gb_metrics = train_model("GradientBoosting", gb, X_train, y_train, X_test, y_test, gb_params)
    results["GradientBoosting"] = (gb_model, gb_metrics)

    # 7. XGBoost (if available)
    if HAS_XGB:
        xgb_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
                      "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42}
        xgb_model_obj = xgb.XGBRegressor(**xgb_params, verbosity=0)
        xgb_model, xgb_metrics = train_model("XGBoost", xgb_model_obj, X_train, y_train, X_test, y_test, xgb_params)
        results["XGBoost"] = (xgb_model, xgb_metrics)

    # 8. Pick best model by MAE
    best_name  = min(results, key=lambda k: results[k][1]["mae"])
    best_model = results[best_name][0]
    best_mae   = results[best_name][1]["mae"]
    print(f"\n[train] ★ Best model: {best_name}  (MAE={best_mae})")

    # 9. Save best model + feature columns
    model_path   = os.path.join(MODELS_DIR, "best_model.pkl")
    feature_path = os.path.join(MODELS_DIR, "feature_cols.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(X_train.columns.tolist(), feature_path)
    print(f"[train] Saved → {model_path}")
    print(f"[train] Saved → {feature_path}")

    return best_model, X_train.columns.tolist()


if __name__ == "__main__":
    run()
