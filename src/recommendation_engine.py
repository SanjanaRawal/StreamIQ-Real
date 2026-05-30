"""
StreamIQ Real – Hybrid Recommendation Engine
=============================================
1. Content-Based  : TF-IDF on genre + decade + language + classic flag
2. Collaborative  : Custom SVD (matrix factorisation via SGD)
3. Hybrid blend   : 40% CB + 60% CF with MinMax normalisation
Real data: 748K ratings, 7K+ movies, 23K users
"""

import os, warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import joblib
warnings.filterwarnings("ignore")

PROC   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS, exist_ok=True)


# ── Content-Based ──────────────────────────────────────────────────────────────

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf   = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.sim_mat = None
        self.content = None

    def _build_tags(self, df: pd.DataFrame) -> pd.Series:
        decade = (df["release_year"] // 10 * 10).astype(int).astype(str) + "s"
        classic= df["is_classic"].map({1:"classic_era",0:"modern_era"})
        lang   = df["language"].str.lower().fillna("english")
        # Repeat genre 3x to boost its weight in TF-IDF
        return (df["genre"].str.lower() + " " + df["genre"].str.lower() + " " +
                df["genre"].str.lower() + " " + decade + " " +
                lang + " " + classic)

    def fit(self, content_df: pd.DataFrame):
        self.content = content_df.reset_index(drop=True)
        tags          = self._build_tags(self.content)
        self.tfidf_mat = self.tfidf.fit_transform(tags)   # sparse – stays small
        self.sim_mat   = None                              # computed on demand
        print(f"  [CB] Sparse matrix: {self.tfidf_mat.shape}, vocab: {len(self.tfidf.vocabulary_)}")
        return self

    def recommend(self, content_id: str, n: int = 10) -> pd.DataFrame:
        idx_ser = self.content.index[self.content["content_id"] == content_id]
        if idx_ser.empty: return pd.DataFrame()
        idx      = idx_ser[0]
        row_vec  = self.tfidf_mat[idx]                    # (1, vocab)
        sims     = cosine_similarity(row_vec, self.tfidf_mat).flatten()
        top_idxs = sims.argsort()[::-1][1:n+1]
        result   = self.content.iloc[top_idxs][["content_id","title","genre",
                                                 "avg_rating","total_views","release_year"]].copy()
        result["similarity_score"] = [round(float(sims[i]), 4) for i in top_idxs]
        return result.reset_index(drop=True)

    def save(self):
        joblib.dump(self, os.path.join(MODELS, "content_based_recommender.pkl"))

    @classmethod
    def load(cls):
        return joblib.load(os.path.join(MODELS, "content_based_recommender.pkl"))


# ── Collaborative Filtering (SVD via SGD) ──────────────────────────────────────

class CollaborativeFilteringRecommender:
    def __init__(self, n_factors=40, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors; self.n_epochs = n_epochs
        self.lr = lr; self.reg = reg
        self.user_map = {}; self.item_map = {}
        self.P = self.Q = self.bu = self.bi = None
        self.global_mean = 0.0

    def fit(self, ratings_df: pd.DataFrame, sample_n: int = 300_000):
        df = ratings_df.dropna(subset=["rating"]).copy()
        if len(df) > sample_n:
            df = df.sample(sample_n, random_state=42)

        self.global_mean = float(df["rating"].mean())
        users = df["user_id"].unique(); items = df["content_id"].unique()
        self.user_map = {u:i for i,u in enumerate(users)}
        self.item_map = {c:i for i,c in enumerate(items)}
        n_u, n_i = len(users), len(items)

        np.random.seed(42)
        self.P  = np.random.normal(0, 0.1, (n_u, self.n_factors))
        self.Q  = np.random.normal(0, 0.1, (n_i, self.n_factors))
        self.bu = np.zeros(n_u); self.bi = np.zeros(n_i)

        records = df[["user_id","content_id","rating"]].values
        print(f"  [CF] Training SVD on {len(records):,} ratings, "
              f"{n_u:,} users, {n_i:,} items …")

        for epoch in range(self.n_epochs):
            np.random.shuffle(records)
            loss = 0.0
            for row in records:
                uid, cid, r = row[0], row[1], float(row[2])
                u = self.user_map.get(uid); i = self.item_map.get(cid)
                if u is None or i is None: continue
                pred = (self.global_mean + self.bu[u] + self.bi[i]
                        + np.dot(self.P[u], self.Q[i]))
                err  = r - pred; loss += err**2
                self.bu[u] += self.lr*(err - self.reg*self.bu[u])
                self.bi[i] += self.lr*(err - self.reg*self.bi[i])
                self.P[u]  += self.lr*(err*self.Q[i] - self.reg*self.P[u])
                self.Q[i]  += self.lr*(err*self.P[u] - self.reg*self.Q[i])
            rmse = np.sqrt(loss / len(records))
            if (epoch+1) % 5 == 0:
                print(f"      Epoch {epoch+1}/{self.n_epochs}  RMSE={rmse:.4f}")

        print(f"  [CF] Done. Final RMSE={rmse:.4f}")
        return self

    def predict(self, user_id, content_id) -> float:
        u = self.user_map.get(user_id); i = self.item_map.get(content_id)
        if u is None or i is None: return self.global_mean
        return float(np.clip(self.global_mean + self.bu[u] + self.bi[i]
                              + np.dot(self.P[u], self.Q[i]), 1, 5))

    def recommend(self, user_id: str, content_df: pd.DataFrame,
                  n: int = 10) -> pd.DataFrame:
        u = self.user_map.get(user_id)
        if u is None:
            return content_df.nlargest(n, "avg_rating")[
                ["content_id","title","genre","avg_rating"]].copy()

        scores = [(cid, float(np.clip(
                    self.global_mean + self.bu[u] + self.bi[i]
                    + np.dot(self.P[u], self.Q[i]), 1, 5)))
                  for cid, i in self.item_map.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_ids  = [s[0] for s in scores[:n]]
        top_pred = {s[0]: round(s[1],3) for s in scores[:n]}
        result   = content_df[content_df["content_id"].isin(top_ids)][
            ["content_id","title","genre","avg_rating","total_views"]].copy()
        result["predicted_rating"] = result["content_id"].map(top_pred)
        return result.sort_values("predicted_rating", ascending=False).reset_index(drop=True)

    def save(self):
        joblib.dump(self, os.path.join(MODELS, "cf_recommender.pkl"))

    @classmethod
    def load(cls):
        return joblib.load(os.path.join(MODELS, "cf_recommender.pkl"))


# ── Hybrid ─────────────────────────────────────────────────────────────────────

class HybridRecommender:
    def __init__(self, cb_weight=0.4, cf_weight=0.6):
        self.cb_weight = cb_weight; self.cf_weight = cf_weight
        self.cb = ContentBasedRecommender()
        self.cf = CollaborativeFilteringRecommender()
        self.content = None

    def fit(self, content_df: pd.DataFrame, interactions_df: pd.DataFrame):
        self.content = content_df
        print("\n[Hybrid] Training Content-Based …")
        self.cb.fit(content_df)
        print("[Hybrid] Training Collaborative Filter …")
        self.cf.fit(interactions_df)
        return self

    def recommend(self, user_id: str, seed_content_id: str = None,
                  n: int = 10) -> pd.DataFrame:
        scaler  = MinMaxScaler()
        cf_recs = self.cf.recommend(user_id, self.content, n=80)

        if "predicted_rating" in cf_recs.columns:
            cf_recs["cf_score"] = scaler.fit_transform(
                cf_recs[["predicted_rating"]].fillna(0))
        else:
            cf_recs["cf_score"] = 0.5

        if seed_content_id:
            cb_recs = self.cb.recommend(seed_content_id, n=80)
            if not cb_recs.empty:
                cb_recs["cb_score"] = scaler.fit_transform(
                    cb_recs[["similarity_score"]].fillna(0))
                merged = cf_recs.merge(
                    cb_recs[["content_id","cb_score"]], on="content_id", how="outer"
                ).fillna(0)
            else:
                merged = cf_recs.copy(); merged["cb_score"] = 0.0
        else:
            merged = cf_recs.copy(); merged["cb_score"] = 0.0

        merged["hybrid_score"] = (self.cf_weight * merged.get("cf_score", 0) +
                                   self.cb_weight * merged.get("cb_score", 0))
        result = merged.sort_values("hybrid_score", ascending=False).head(n)
        keep   = [c for c in ["content_id","title","genre","avg_rating",
                               "total_views","hybrid_score"] if c in result.columns]
        return result[keep].reset_index(drop=True)

    def save(self):
        joblib.dump(self, os.path.join(MODELS, "hybrid_recommender.pkl"))
        print("[Hybrid] Model saved.")

    @classmethod
    def load(cls):
        return joblib.load(os.path.join(MODELS, "hybrid_recommender.pkl"))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("  STREAMIQ REAL – RECOMMENDATION ENGINE")
    print("="*55)

    content      = pd.read_csv(os.path.join(PROC, "content_catalog.csv"))
    interactions = pd.read_csv(os.path.join(PROC, "interactions.csv"))

    hybrid = HybridRecommender(cb_weight=0.4, cf_weight=0.6)
    hybrid.fit(content, interactions)
    hybrid.save()

    # Demo
    uid = interactions["user_id"].iloc[100]
    cid = interactions["content_id"].iloc[100]
    print(f"\n📌 Demo for user={uid}, seed={cid}")
    recs = hybrid.recommend(uid, cid, n=5)
    print(recs[["title","genre","avg_rating","hybrid_score"]].to_string(index=False))
    print("\n✅ Recommendation engine ready!")


if __name__ == "__main__":
    main()
