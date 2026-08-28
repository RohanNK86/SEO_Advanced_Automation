"""
RandomForest Classifier — Content Action Prediction
FlyRank Content Intelligence Platform

Predicts action label (Refresh Content / Review / No Action)
from SEO features.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from feature_pipeline.engineer import run as engineer_run
from feature_pipeline.leakage_audit import ALLOWED_INPUT_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ACTION_COL = "action"


def run():
    df = engineer_run()

    # Encode categoricals
    feature_cols = [f for f in ALLOWED_INPUT_FEATURES if f in df.columns]
    X = df[feature_cols].copy()
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    le_action = LabelEncoder()
    y = le_action.fit_transform(df[ACTION_COL].fillna("Review"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    rfc = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rfc.fit(X_train_sc, y_train)

    preds = rfc.predict(X_test_sc)
    acc = accuracy_score(y_test, preds)
    print(f"\n[TreeClassify] Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=le_action.classes_))

    # Save
    joblib.dump(rfc,       os.path.join(MODELS_DIR, "action_classifier.pkl"))
    joblib.dump(scaler,    os.path.join(MODELS_DIR, "action_scaler.pkl"))
    joblib.dump(le_action, os.path.join(MODELS_DIR, "action_label_encoder.pkl"))
    joblib.dump(X_train.columns.tolist(), os.path.join(MODELS_DIR, "classifier_features.pkl"))
    print("[TreeClassify] Models saved.")
    return rfc, scaler, le_action


if __name__ == "__main__":
    run()