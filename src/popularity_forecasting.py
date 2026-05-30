"""
StreamIQ Real – Popularity Forecasting
========================================
1. Holt-Winters Exponential Smoothing (season_len=4 for monthly = quarterly cycle)
2. GBM on lag features
3. Ensemble blend + trend scoring
"""

import os, warnings
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import joblib
warnings.filterwarnings("ignore")

PROC   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS, exist_ok=True)


# ── Lag features ───────────────────────────────────────────────────────────────

def make_lag_features(series: pd.Series, lags=(1,2,3,4,6,12)) -> pd.DataFrame:
    df = pd.DataFrame({"y": series.values})
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["rolling_3_mean"] = df["y"].shift(1).rolling(3).mean()
    df["rolling_6_mean"] = df["y"].shift(1).rolling(6).mean()
    df["rolling_3_std"]  = df["y"].shift(1).rolling(3).std()
    df["month_num"]      = np.arange(len(series)) % 12
    df["quarter"]        = df["month_num"] // 3
    return df.dropna()


# ── Holt-Winters ───────────────────────────────────────────────────────────────

class HoltWintersForecaster:
    def __init__(self, alpha=0.3, beta=0.1, gamma=0.2, season_len=4):
        self.alpha = alpha; self.beta = beta
        self.gamma = gamma; self.season_len = season_len
        self.level_ = self.trend_ = None
        self.seasonal_ = []; self.history_ = None

    def fit(self, series: np.ndarray):
        m = self.season_len
        if len(series) < 2 * m:
            m = max(2, len(series) // 2)
            self.season_len = m

        L = np.mean(series[:m])
        T = (np.mean(series[m:2*m]) - np.mean(series[:m])) / m if len(series) >= 2*m else 0
        S = [series[i] / max(L, 1e-6) for i in range(m)]
        level, trend = L, T

        for t in range(m, len(series)):
            y_t  = series[t]; s_t = S[t % m]
            L_new = self.alpha*(y_t/max(s_t,1e-6)) + (1-self.alpha)*(level+trend)
            T_new = self.beta*(L_new-level) + (1-self.beta)*trend
            S[t%m]= self.gamma*(y_t/max(L_new,1e-6)) + (1-self.gamma)*s_t
            level, trend = L_new, T_new

        self.level_ = level; self.trend_ = trend
        self.seasonal_ = S; self.history_ = series.copy()
        return self

    def forecast(self, steps: int) -> np.ndarray:
        m = self.season_len
        return np.array([
            max(0, (self.level_ + h*self.trend_) * self.seasonal_[(len(self.history_)+h-1)%m])
            for h in range(1, steps+1)])


# ── ML Forecaster ──────────────────────────────────────────────────────────────

class MLForecaster:
    def __init__(self):
        self.model  = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42)
        self.scaler = StandardScaler()
        self.lags   = (1,2,3,4,6,12)

    def fit(self, series: pd.Series):
        feat = make_lag_features(series, self.lags)
        X    = feat.drop("y", axis=1).values
        y    = feat["y"].values
        Xs   = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.history_ = series.copy()
        return self

    def forecast(self, steps: int, series: pd.Series) -> np.ndarray:
        history = list(series.values)
        preds   = []
        for step in range(steps):
            row = {}
            for lag in self.lags:
                row[f"lag_{lag}"] = history[-lag] if len(history) >= lag else 0
            row["rolling_3_mean"] = np.mean(history[-3:]) if len(history)>=3 else np.mean(history)
            row["rolling_6_mean"] = np.mean(history[-6:]) if len(history)>=6 else np.mean(history)
            row["rolling_3_std"]  = np.std(history[-3:])  if len(history)>=3 else 0
            row["month_num"]      = (len(history)+step) % 12
            row["quarter"]        = row["month_num"] // 3
            X  = np.array([[row[k] for k in sorted(row)]])
            Xs = self.scaler.transform(X)
            p  = max(0, float(self.model.predict(Xs)[0]))
            preds.append(p); history.append(p)
        return np.array(preds)


# ── Orchestrator ───────────────────────────────────────────────────────────────

class PopularityForecaster:
    def __init__(self):
        self.hw_models = {}; self.ml_models = {}
        self.ts_data_  = None

    def fit(self, ts_df: pd.DataFrame, top_n: int = 50):
        ts_df = ts_df.copy()
        ts_df["date"] = pd.to_datetime(ts_df["date"])
        self.ts_data_ = ts_df

        top_ids = (ts_df.groupby("content_id")["daily_views"]
                       .sum().nlargest(top_n).index.tolist())
        print(f"  Fitting forecasters for {len(top_ids)} titles …")

        for cid in top_ids:
            sub  = ts_df[ts_df["content_id"]==cid].sort_values("date")
            vals = sub["daily_views"].values.astype(float)
            s    = pd.Series(vals, index=pd.to_datetime(sub["date"]))

            if len(vals) < 4:
                continue

            hw = HoltWintersForecaster(season_len=min(4, len(vals)//2))
            hw.fit(vals); self.hw_models[cid] = hw

            ml = MLForecaster()
            ml.fit(s); self.ml_models[cid] = ml

        print(f"  Done. {len(self.hw_models)} forecasters fitted.")
        return self

    def forecast(self, content_id: str, steps: int = 12,
                 method: str = "ensemble") -> pd.DataFrame:
        last = self.ts_data_["date"].max()
        dates= [last + pd.DateOffset(months=i+1) for i in range(steps)]

        if content_id not in self.hw_models:
            return pd.DataFrame({"date":dates,"forecast":[0]*steps})

        sub    = self.ts_data_[self.ts_data_["content_id"]==content_id].sort_values("date")
        series = pd.Series(sub["daily_views"].values.astype(float),
                           index=pd.to_datetime(sub["date"]))

        hw_p = self.hw_models[content_id].forecast(steps)
        ml_p = self.ml_models[content_id].forecast(steps, series)

        preds = {"hw": hw_p, "ml": ml_p}.get(method, 0.5*hw_p + 0.5*ml_p)
        return pd.DataFrame({
            "date":    dates, "forecast": np.round(preds).astype(int),
            "hw_pred": np.round(hw_p).astype(int),
            "ml_pred": np.round(ml_p).astype(int),
        })

    def trending_score(self, content_id: str, window: int = 6) -> float:
        sub = self.ts_data_[self.ts_data_["content_id"]==content_id].tail(window)
        if len(sub) < 3: return 0.0
        y = sub["daily_views"].values.astype(float)
        slope, *_ = stats.linregress(np.arange(len(y)), y)
        return round(float(slope / (np.mean(y)+1e-6)), 4)

    def get_trending(self, top_n: int = 15) -> pd.DataFrame:
        scores = {cid: self.trending_score(cid) for cid in self.hw_models}
        df = pd.DataFrame(scores.items(), columns=["content_id","trend_score"])
        return df.sort_values("trend_score", ascending=False).head(top_n)

    def save(self):
        joblib.dump(self, os.path.join(MODELS, "popularity_forecaster.pkl"))

    @classmethod
    def load(cls):
        return joblib.load(os.path.join(MODELS, "popularity_forecaster.pkl"))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("  STREAMIQ REAL – POPULARITY FORECASTING")
    print("="*55)

    ts      = pd.read_csv(os.path.join(PROC, "popularity_ts.csv"))
    content = pd.read_csv(os.path.join(PROC, "content_catalog.csv"))

    print(f"\n  Timeseries: {len(ts):,} rows, "
          f"{ts['content_id'].nunique()} titles, "
          f"date range {ts['date'].min()[:7]} → {ts['date'].max()[:7]}")

    fc = PopularityForecaster()
    fc.fit(ts, top_n=50)

    # Demo
    sample_id = list(fc.hw_models.keys())[0]
    demo = fc.forecast(sample_id, steps=6)
    print(f"\n  6-month forecast for {sample_id}:")
    print(demo.to_string(index=False))

    # Trending
    trending = fc.get_trending(15)
    trending = trending.merge(
        content[["content_id","title","genre"]], on="content_id", how="left")
    trending.to_csv(os.path.join(PROC, "trending_content.csv"), index=False)
    print("\n  Top Trending:")
    print(trending.head(5).to_string(index=False))

    fc.save()
    print("\n✅ Popularity forecasting complete!")


if __name__ == "__main__":
    main()
