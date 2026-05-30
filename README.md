# 🎬 StreamIQ Real — AI-Enhanced Streaming Analytics (Real Data Edition)

> End-to-end data science portfolio project built on **real MovieTweetings + IMDB data**.
> 921K ratings · 38K movies · 23K users · 4 AI/ML modules · Interactive Streamlit dashboard.

---

## 🗂 Project Structure

```
StreamIQ_Real/
├── data/
│   ├── raw/                          ← Downloaded source files (auto-fetched)
│   │   ├── ratings.dat               MovieTweetings: 921K ratings
│   │   ├── movies.dat                MovieTweetings: 38K movie titles
│   │   └── imdb_genres.csv           IMDB: runtime, score, director, budget
│   ├── processed/                    ← Cleaned outputs (auto-generated)
│   │   ├── content_catalog.csv       7,316 enriched movie records
│   │   ├── interactions.csv          748K cleaned interactions
│   │   ├── users.csv                 23K behavioural user profiles
│   │   ├── popularity_ts.csv         Monthly rating counts (top 80 movies)
│   │   ├── sentiment_results.csv     Per-review sentiment scores
│   │   ├── content_sentiment_agg.csv Per-movie sentiment aggregates
│   │   ├── user_features.csv         Engineered ML feature matrix
│   │   ├── trending_content.csv      Trend-scored movies
│   │   └── model_metrics.csv         Evaluation results
│   └── ingest_and_clean.py           ← Data pipeline (fetch + clean + transform)
│
├── src/
│   ├── recommendation_engine.py      Hybrid SVD + TF-IDF recommender
│   ├── sentiment_analysis.py         TextBlob + domain lexicon + aspect mining
│   ├── behaviour_prediction.py       Churn RF · Watch GBM · Genre LogReg
│   └── popularity_forecasting.py     Holt-Winters + GBM ensemble forecaster
│
├── models/                           ← Saved .pkl artefacts (auto-generated)
│   ├── hybrid_recommender.pkl
│   ├── churn_predictor.pkl
│   ├── watch_completion_predictor.pkl
│   ├── next_genre_predictor.pkl
│   ├── popularity_forecaster.pkl
│   └── sentiment_lexicon.pkl
│
├── dashboard/
│   └── app.py                        ← Streamlit dashboard (5 pages)
│
├── notebooks/
│   └── EDA.py                        ← Full exploratory analysis + 4 plots
│
├── reports/                          ← EDA plots saved here
│
├── train_all.py                      ← One command trains everything
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

### Step 2 — Train all models
```bash
python train_all.py
```

This automatically:
- Downloads MovieTweetings ratings + movies from GitHub (~27 MB)
- Downloads IMDB movie stats from GitHub (~1.3 MB)
- Cleans and transforms all raw data
- Trains all 5 ML models
- Saves everything to `data/processed/` and `models/`

Total time: ~5–8 minutes depending on machine.

### Step 3 — Launch dashboard
```bash
streamlit run dashboard/app.py
```

### Optional — Run EDA
```bash
python notebooks/EDA.py
# Saves 4 dark-themed plots to reports/
```

---

## 📦 Data Sources

### MovieTweetings
- **Source**: [github.com/sidooms/MovieTweetings](https://github.com/sidooms/MovieTweetings)
- **What it is**: Real ratings extracted from structured movie tweets on Twitter, linked to IMDb
- **Size**: 921,398 ratings by ~70K users on ~38K movies
- **Format**: `user_id::imdb_id::rating::timestamp` (ratings 1–10, mapped to 1–5)
- **License**: MIT — free for research and portfolio use

### IMDB Movie Stats
- **Source**: [github.com/danielgrijalva/movie-stats](https://github.com/danielgrijalva/movie-stats)
- **What it is**: IMDB metadata — runtime, director, country, budget, gross, score
- **Size**: 7,668 movies
- **Used for**: Enriching content catalog with runtime, director info

### After Cleaning
| Metric | Raw | Cleaned |
|---|---|---|
| Ratings | 921,398 | 748,540 (81% retained) |
| Movies | 38,018 | 7,316 |
| Users | ~70,000 | 23,058 |
| Rating scale | 1–10 | 1–5 (halved) |
| Year range | 1900–2024 | 1980–2024 |

**Cleaning steps applied:**
- Removed users with fewer than 5 ratings
- Removed movies with fewer than 10 ratings
- Removed duplicate user-movie pairs (kept latest)
- Filtered movies outside 1980–2024
- Normalised genre names (39 IMDB genres → 15 clean categories)
- Scaled ratings from 1–10 to 1–5

---

## 🧠 AI Modules

### 1. Recommendation Engine
**File**: `src/recommendation_engine.py`

| Component | Detail |
|---|---|
| Content-Based | TF-IDF on `genre × genre × genre + decade + language + classic_flag` (genre repeated 3× to boost weight) |
| Collaborative | Custom SGD-based SVD, 40 latent factors, 20 epochs, trained on 300K ratings |
| Hybrid | 60% CF + 40% CB with MinMax normalisation |
| **RMSE** | **0.614** on 1–5 scale (real signal from real user ratings) |
| Cold-start | CB-only for new users; global mean fallback |

### 2. Sentiment Analysis
**File**: `src/sentiment_analysis.py`

| Component | Detail |
|---|---|
| TextBlob | Sentence-level polarity (−1 to +1) + subjectivity scoring |
| Domain lexicon | 40 streaming/movie-specific terms (masterpiece +2.0, boring −1.5, etc.) |
| Combination | `0.6 × TextBlob + 0.4 × lexicon`, clipped to [−1, 1] |
| Aspect mining | Detects acting / plot / visuals / sound mentions, scores each independently |
| Reviews | Generated from rating × genre (proxy for real reviews; real scraped reviews would plug in directly) |

### 3. Behaviour Prediction
**File**: `src/behaviour_prediction.py`

| Model | Algorithm | Metric | Value |
|---|---|---|---|
| Churn Predictor | Random Forest (200 trees, class-balanced) | AUC-ROC | **1.000** |
| Watch Completion | Gradient Boosting (200 trees, LR=0.05) | R² / MAE | **0.596 / 7.95%** |
| Next-Genre | Logistic Regression (L2, lbfgs) | Accuracy | **32.9%** (vs 6.7% random) |

> **Note on churn AUC=1.0**: Churn is derived from `days_inactive > 180`. Since `days_since_last` is a direct feature, the model perfectly separates churn from non-churn. In production, churn labels come from a separate ground-truth source, making this a genuinely hard problem. The architecture is production-ready — the label definition makes this trivial here.

### 4. Popularity Forecasting
**File**: `src/popularity_forecasting.py`

| Component | Detail |
|---|---|
| Holt-Winters | Triple exponential smoothing, quarterly seasonality (season_len=4 months) |
| GBM | Lag features: t−1,2,3,4,6,12 months + rolling stats + month/quarter |
| Ensemble | 50/50 blend with ±12% confidence band |
| Trend score | `slope / mean` over trailing 6 months |
| Coverage | Top 50 movies by total rating volume |

---

## 📊 Key Data Insights

- **Long-tail distribution**: Top 1% of users contribute ~40% of ratings — classic power law
- **Matrix sparsity**: 99.56% of the user-movie matrix is unobserved — makes CF challenging and realistic
- **Genre bias**: Animation and Drama receive the highest average ratings; Horror the lowest
- **Temporal pattern**: Rating activity peaks in evening hours (18:00–22:00); weekends spike
- **Rating bias**: Most users rate 3.0–4.0 on average; very few harsh (<2.5) or generous (>4.5) raters

---

## 🚢 Deployment (Free)

### Streamlit Community Cloud
```bash
# 1. Push to GitHub (include models/ and data/processed/)
git init && git add . && git commit -m "StreamIQ Real"
git remote add origin https://github.com/USERNAME/StreamIQ_Real.git
git push -u origin main

# 2. Go to share.streamlit.io → New App
#    Main file: dashboard/app.py
```

### Hugging Face Spaces (16GB RAM free)
```bash
# Create a Space with SDK: Streamlit
# Clone, copy files in, push
git clone https://huggingface.co/spaces/USERNAME/StreamIQ_Real
cp -r StreamIQ_Real/* StreamIQ_Real_hf/
cd StreamIQ_Real_hf && git add . && git commit -m "deploy" && git push
```

> **Important**: Commit `models/*.pkl` and `data/processed/*.csv` to the repo.
> The cloud cannot run `train_all.py` at deploy time.
> If `.pkl` files exceed 100MB, use `git lfs track "*.pkl"`.

---

## 🛠 Tech Stack

```
Python 3.10+      Core language
Pandas / NumPy    Data manipulation & feature engineering
Scikit-learn      ML models, pipelines, preprocessing
TextBlob / NLTK   NLP sentiment analysis
Plotly            Interactive visualisations
Streamlit         Dashboard framework
SciPy             Statistical utilities (trend scoring)
Joblib            Model serialisation
urllib            Data downloading (no extra dependencies)
```

---

## 💼 Interview Talking Points

**On the data pipeline:**
> "I built a full ingestion pipeline that downloads MovieTweetings from GitHub, applies quality filters — minimum 5 ratings per user, 10 per movie — maps 39 IMDB genres to 15 clean categories, and derives user profiles purely from behavioural signals since we have no demographic data."

**On the recommendation engine:**
> "I implemented SVD from scratch using SGD — understanding the math, not just calling fit(). With 40 latent factors trained on 300K ratings, I achieved RMSE 0.614 on a 1–5 scale, which is competitive with published results on similar datasets."

**On sparsity:**
> "The user-movie matrix is 99.56% sparse — a real cold-start challenge. I handled it with a content-based fallback for new users and regularisation in the SVD to prevent overfitting to the few observed entries."

**On the sentiment approach:**
> "Since MovieTweetings has no text reviews, I used a domain-specific lexicon layer on top of TextBlob, weighted toward streaming/movie vocabulary. A BERT fine-tuned on movie reviews would be the next upgrade."

---

*StreamIQ Real — Portfolio Data Science Project*
*Real data · Real models · Real insights*
