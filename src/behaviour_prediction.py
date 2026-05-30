"""
StreamIQ Real – Viewer Behaviour Prediction
=============================================
1. Churn Predictor          (Random Forest)
2. Watch-Completion Regressor (Gradient Boosting)
3. Next-Genre Predictor     (Logistic Regression)
All trained on real MovieTweetings-derived features.
"""

import os, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                              mean_absolute_error, r2_score)
import joblib
warnings.filterwarnings("ignore")

PROC   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS, exist_ok=True)


# ── Feature Engineering ────────────────────────────────────────────────────────

def build_user_features(users_df: pd.DataFrame,
                        interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Rich per-user feature matrix from real behavioural data."""
    df = interactions_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    agg = df.groupby("user_id").agg(
        total_watches    = ("content_id",       "count"),
        avg_watch_pct    = ("watch_percentage",  "mean"),
        unique_genres    = ("genre",            "nunique"),
        avg_rating_given = ("rating",            "mean"),
        num_ratings      = ("rating",            "count"),
        mobile_sessions  = ("device",           lambda x: (x=="Mobile").sum()),
        tv_sessions      = ("device",           lambda x: (x=="Smart TV").sum()),
        evening_watches  = ("hour",             lambda x: ((x>=18)|(x<=2)).sum()),
        weekend_watches  = ("dayofweek",        lambda x: (x>=5).sum()),
        latest_watch     = ("timestamp",        "max"),
    ).reset_index()

    max_date = df["timestamp"].max()
    agg["days_since_last"]  = (max_date - agg["latest_watch"]).dt.days
    agg["rating_ratio"]     = agg["num_ratings"]  / (agg["total_watches"] + 1)
    agg["mobile_ratio"]     = agg["mobile_sessions"] / (agg["total_watches"] + 1)
    agg["evening_ratio"]    = agg["evening_watches"]  / (agg["total_watches"] + 1)
    agg["weekend_ratio"]    = agg["weekend_watches"]  / (agg["total_watches"] + 1)

    users = users_df.copy()
    users["first_seen"] = pd.to_datetime(users["first_seen"])
    users["last_seen"]  = pd.to_datetime(users["last_seen"])

    # Drop columns that also exist in agg to prevent _x/_y suffix collision
    overlap = [c for c in agg.columns if c != "user_id" and c in users.columns]
    users = users.drop(columns=overlap)

    # Encode categoricals
    le_age = LabelEncoder(); le_sub = LabelEncoder()
    le_risk= LabelEncoder(); le_gen = LabelEncoder()
    le_dev = LabelEncoder()

    users["age_enc"]      = le_age.fit_transform(users["age_group"])
    users["sub_enc"]      = le_sub.fit_transform(users["subscription"])
    users["risk_enc"]     = le_risk.fit_transform(users["churn_risk"])
    users["genre_enc"]    = le_gen.fit_transform(users["favorite_genre"])
    users["dev_enc"]      = le_dev.fit_transform(users["favorite_device"])
    users["is_active_bin"]= users["is_active"].astype(int)

    merged = users.merge(agg, on="user_id", how="left").fillna(0)
    return merged


def build_watch_features(interactions_df: pd.DataFrame,
                         content_df: pd.DataFrame) -> pd.DataFrame:
    inter_slim = interactions_df.drop(columns=["genre"], errors="ignore")
    df = inter_slim.merge(
        content_df[["content_id","genre","release_year","avg_rating",
                    "runtime_min","is_original","is_classic"]],
        on="content_id", how="left")

    df["timestamp"]  = pd.to_datetime(df["timestamp"])
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_evening"] = ((df["hour"] >= 18) | (df["hour"] <= 2)).astype(int)

    le_genre  = LabelEncoder()
    le_device = LabelEncoder()
    le_mood   = LabelEncoder()
    df["genre_enc"]  = le_genre.fit_transform(df["genre"].fillna("Unknown"))
    df["device_enc"] = le_device.fit_transform(df["device"].fillna("Unknown"))
    df["mood_enc"]   = le_mood.fit_transform(df["mood"].fillna("Unknown"))
    df["is_original"]= df["is_original"].astype(int)
    df["is_classic"] = df["is_classic"].fillna(0).astype(int)
    return df


# ── 1. Churn Predictor ─────────────────────────────────────────────────────────

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = [
            "age_enc","sub_enc","tenure_days","avg_daily_hours",
            "total_watches","avg_watch_pct","unique_genres",
            "avg_rating_given","num_ratings","rating_ratio",
            "mobile_ratio","evening_ratio","weekend_ratio",
            "days_since_last","genre_enc","dev_enc","is_active_bin"
        ]

    def fit(self, user_features_df: pd.DataFrame) -> dict:
        df = user_features_df.copy()
        df["churn_binary"] = (df["churn_risk"] == "High").astype(int)

        avail = [c for c in self.feature_cols if c in df.columns]
        X = df[avail].fillna(0)
        y = df["churn_binary"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=8,
            class_weight="balanced", random_state=42, n_jobs=-1)
        self.model.fit(X_tr, y_tr)

        preds = self.model.predict(X_te)
        proba = self.model.predict_proba(X_te)[:, 1]
        auc   = roc_auc_score(y_te, proba)
        rep   = classification_report(y_te, preds, output_dict=True)
        fi    = pd.DataFrame({"feature": avail,
                               "importance": self.model.feature_importances_}
                             ).sort_values("importance", ascending=False)

        print(f"  [Churn] AUC-ROC : {auc:.4f}")
        print(f"  [Churn] Accuracy: {rep['accuracy']:.4f}")
        print(f"  [Churn] Top features: {fi['feature'].head(3).tolist()}")
        return {"auc": auc, "report": rep, "feature_importance": fi}

    def save(self):
        joblib.dump(self, os.path.join(MODELS, "churn_predictor.pkl"))

    @classmethod
    def load(cls):
        return joblib.load(os.path.join(MODELS, "churn_predictor.pkl"))


# ── 2. Watch Completion Predictor ─────────────────────────────────────────────

class WatchCompletionPredictor:
    def __init__(self):
        self.model = None
        self.feature_cols = [
            "genre_enc","device_enc","mood_enc","hour","dayofweek",
            "is_weekend","is_evening","avg_rating","runtime_min",
            "is_original","is_classic","release_year","rating"
        ]

    def fit(self, watch_features_df: pd.DataFrame) -> dict:
        avail = [c for c in self.feature_cols if c in watch_features_df.columns]
        df    = watch_features_df[avail + ["watch_percentage"]].dropna()

        # Sample for speed (748K rows is large)
        if len(df) > 200_000:
            df = df.sample(200_000, random_state=42)

        X = df[avail].fillna(0)
        y = df["watch_percentage"]

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42)
        self.model.fit(X_tr, y_tr)

        preds = self.model.predict(X_te)
        mae   = mean_absolute_error(y_te, preds)
        r2    = r2_score(y_te, preds)
        fi    = pd.DataFrame({"feature": avail,
                               "importance": self.model.feature_importances_}
                             ).sort_values("importance", ascending=False)
        print(f"  [Watch] MAE={mae:.2f}%  R²={r2:.4f}")
        return {"mae": mae, "r2": r2, "feature_importance": fi}

    def save(self):
        joblib.dump(self, os.path.join(MODELS, "watch_completion_predictor.pkl"))

    @classmethod
    def load(cls):
        return joblib.load(os.path.join(MODELS, "watch_completion_predictor.pkl"))


# ── 3. Next-Genre Predictor ───────────────────────────────────────────────────

class NextGenrePredictor:
    def __init__(self):
        self.pipeline = None
        self.le = LabelEncoder()

    def fit(self, interactions_df: pd.DataFrame) -> dict:
        df = interactions_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["user_id", "timestamp"])
        df["prev_genre"] = df.groupby("user_id")["genre"].shift(1)
        df = df.dropna(subset=["prev_genre", "genre"])

        # Sample for speed
        if len(df) > 150_000:
            df = df.sample(150_000, random_state=42)

        le_prev = LabelEncoder()
        df["prev_genre_enc"] = le_prev.fit_transform(df["prev_genre"])
        X = df[["prev_genre_enc","hour","dayofweek","watch_percentage",
                "rating"]].fillna(0)
        y = self.le.fit_transform(df["genre"])

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=500, C=1.0,
                                           solver="lbfgs", random_state=42)),
        ])
        self.pipeline.fit(X_tr, y_tr)
        acc = self.pipeline.score(X_te, y_te)
        print(f"  [Genre] Accuracy={acc:.4f}  Classes={len(self.le.classes_)}")
        return {"accuracy": acc, "classes": self.le.classes_.tolist()}

    def save(self):
        joblib.dump(self, os.path.join(MODELS, "next_genre_predictor.pkl"))

    @classmethod
    def load(cls):
        return joblib.load(os.path.join(MODELS, "next_genre_predictor.pkl"))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("  STREAMIQ REAL – BEHAVIOUR PREDICTION")
    print("="*55)

    users        = pd.read_csv(os.path.join(PROC, "users.csv"))
    interactions = pd.read_csv(os.path.join(PROC, "interactions.csv"))
    content      = pd.read_csv(os.path.join(PROC, "content_catalog.csv"))

    print("\n[1/3] Building feature matrices …")
    user_feat  = build_user_features(users, interactions)
    watch_feat = build_watch_features(interactions, content)
    user_feat.to_csv(os.path.join(PROC, "user_features.csv"), index=False)

    print("\n[2/3] Training Churn Predictor …")
    churn = ChurnPredictor()
    churn_m = churn.fit(user_feat)
    churn.save()

    print("\n[3/3] Training Watch Completion Predictor …")
    watch = WatchCompletionPredictor()
    watch_m = watch.fit(watch_feat)
    watch.save()

    print("\n[+] Training Next-Genre Predictor …")
    genre = NextGenrePredictor()
    genre_m = genre.fit(interactions)
    genre.save()

    metrics = pd.DataFrame([
        {"model":"Churn Predictor",   "metric":"AUC-ROC",  "value": churn_m["auc"]},
        {"model":"Watch Completion",  "metric":"R²",       "value": watch_m["r2"]},
        {"model":"Watch Completion",  "metric":"MAE (%)",  "value": watch_m["mae"]},
        {"model":"Next Genre",        "metric":"Accuracy", "value": genre_m["accuracy"]},
    ])
    metrics.to_csv(os.path.join(PROC, "model_metrics.csv"), index=False)
    print("\n" + metrics.to_string(index=False))
    print("\n✅ All behaviour models trained & saved!")


if __name__ == "__main__":
    main()
