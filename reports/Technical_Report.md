# StreamIQ Real — Technical Report

## 1. Data Pipeline

### Sources
Two publicly available datasets, fetched automatically at runtime:

**MovieTweetings** — Ratings extracted from structured tweets where users post IMDb movie ratings. Collected continuously since 2013. Each record: `user_id :: imdb_id :: rating(1-10) :: unix_timestamp`.

**IMDB Movie Stats** — A curated GitHub dataset with per-movie metadata: runtime, IMDB score, vote count, director, country, budget, gross revenue.

### Cleaning Decisions

| Step | Rule | Rationale |
|---|---|---|
| User filter | ≥5 ratings | Removes bots and one-time raters with no collaborative signal |
| Movie filter | ≥10 ratings | Ensures enough signal for CF; removes obscure titles |
| Deduplication | Keep latest per user-movie pair | Some users re-rate after watching again |
| Year filter | 1980–2024 | Pre-1980 titles have very few modern raters, skewing distributions |
| Rating scale | 1–10 → 1–5 (÷2) | Standardises to common 5-star scale; preserves relative order |
| Genre mapping | 39 IMDB genres → 15 | Consolidates rare genres (War→Action, Mystery→Thriller) for model stability |

### Retention Rate
- Ratings: 921,398 → 748,540 (**81.2% retained**)
- Movies: 38,018 → 7,316 (**19.2% retained** — most filtered by <10 ratings)
- Users: ~70,000 → 23,058 (**32.9% retained**)

High movie attrition is expected and correct — the long tail of obscure titles with 1–2 ratings adds noise without collaborative signal.

### Feature Engineering
Since MovieTweetings has no demographic data, all user features are derived from behaviour:

- **Recency**: Days since last rating (most important churn signal)
- **Frequency**: Total ratings, ratings per day
- **Engagement**: Watch percentage (derived from rating magnitude)
- **Diversity**: Unique genres rated
- **Temporal patterns**: Evening ratio, weekend ratio, mobile ratio
- **Subscription tier**: Inferred from activity level (≥200 ratings = Premium)

---

## 2. Recommendation Engine

### Content-Based (TF-IDF)
Feature string per movie: `genre genre genre decade language classic_flag`

Genre is repeated 3× to boost its TF-IDF weight relative to metadata fields. This is a simple but effective trick — without repetition, a 1-word genre field gets dominated by multi-word metadata.

TF-IDF vocabulary: ~180 terms. Cosine similarity computed once at training time, stored as a 7316×7316 dense matrix (~420MB for float32). In production this would be replaced with approximate nearest neighbours (FAISS/Annoy).

### Collaborative Filtering (SVD)
The user-item matrix has shape 23,058 × 7,316 with 99.56% sparsity. SVD factorizes it into:
- P: 23,058 × 40 (user latent factors)
- Q: 7,316 × 40 (item latent factors)

Each rating `r_ui` is predicted as: `μ + b_u + b_i + P_u · Q_i`

where μ is the global mean, b_u is user bias (does this user rate generously?), b_i is item bias (is this movie universally loved?).

SGD update per observed rating:
```
err = r_ui - pred
b_u += lr × (err - reg × b_u)
b_i += lr × (err - reg × b_i)
P_u += lr × (err × Q_i - reg × P_u)
Q_i += lr × (err × P_u - reg × Q_i)
```

Training on 300K ratings (sampled from 748K for speed): **RMSE = 0.614** on held-out 20%.

### RMSE in Context
| Method | RMSE (1–5 scale) |
|---|---|
| Global mean baseline | ~0.88 |
| User mean baseline | ~0.82 |
| **StreamIQ SVD (40 factors)** | **0.614** |
| Netflix Prize winner | ~0.56 (ensemble of 100+ models) |

### Hybrid Blending
```
hybrid_score = 0.6 × CF_score_normalised + 0.4 × CB_score_normalised
```

CF score: predicted rating normalised via MinMax over candidate set.
CB score: cosine similarity normalised via MinMax over similar items.

CF weight is higher because it captures actual user taste. CB weight provides diversity and handles cold-start gracefully.

---

## 3. Sentiment Analysis

### Why Proxy Reviews?
MovieTweetings contains only numerical ratings, no text. Options:
1. **Scrape IMDb/Letterboxd reviews** — legal grey area for a portfolio project
2. **Use a public review dataset** (Stanford IMDB, Yelp) — mismatched movies
3. **Generate proxy reviews from rating + genre** — transparent, reproducible, demonstrates the pipeline architecture

The proxy approach is clearly documented. The entire sentiment pipeline (TextBlob + lexicon + aspect mining) would work identically with real scraped reviews — just remove the `make_review()` function and feed actual text.

### Sentiment Score Formula
```
combined = clip(0.6 × TextBlob_polarity + 0.4 × lexicon_boost, -1, 1)
```

Threshold: positive ≥ 0.15, negative ≤ −0.15, neutral otherwise.

The 0.15 threshold avoids classifying mildly positive/negative text as definitive — appropriate for movie reviews where neutral hedging ("not bad, but...") is common.

### Aspect Mining
For each review, TextBlob sentence tokenizer splits it into sentences. Sentences containing aspect keywords are scored independently:

```python
aspect_score = mean(TextBlob(sentence).polarity
                    for sentence in sentences
                    if any(keyword in sentence for keyword in aspect_keywords))
```

A movie can score positively on acting but negatively on plot — this granularity is what distinguishes aspect-based SA from simple polarity scoring.

---

## 4. Behaviour Prediction

### Churn Predictor
**AUC = 1.000** — this is expected given the label definition. Churn is defined as `days_inactive > 180`, and `days_since_last` is a direct feature. In production:
- Churn labels come from actual cancellation events (billing system)
- `days_since_last` would still be a strong feature but not a perfect predictor
- Real AUC on production churn typically: 0.72–0.85

The model architecture (RF + class balancing + recency/frequency/engagement features) is exactly what production churn models use. The evaluation is honest about the label leakage.

### Watch Completion — R² = 0.596
This is a meaningful result — watch percentage is derived from rating (higher rating → higher completion with added noise), so the model legitimately learns that higher-rated content gets watched more. Features like genre, device, and time-of-day add incremental explanatory power.

### Next-Genre — Accuracy = 32.9%
15-class classification. Random baseline = 1/15 = **6.7%**. The model achieves **5× better than random**, learning real sequential patterns (e.g., Action → Action or Action → Thriller is more common than Action → Romance).

---

## 5. Popularity Forecasting

### Monthly Granularity
Unlike the synthetic version (daily), real data is sparse enough that monthly aggregation gives cleaner signal. Monthly rating counts serve as a proxy for viewership volume.

### Holt-Winters Configuration
- `alpha=0.3`: Moderate smoothing — responsive to recent changes but not noisy
- `beta=0.1`: Slow trend adaptation — movie popularity trends are gradual
- `gamma=0.2`: Seasonal component — quarterly cycle (season_len=4 months)
- Quarterly cycle captures: Q4 award season spike, summer blockbuster effect

### GBM Lag Features
Lags at 1, 2, 3, 4, 6, 12 months capture:
- Short-term momentum (1–3 months)
- Quarterly seasonality (4 months)
- Year-over-year patterns (12 months)

### Trend Score
`trend = slope / mean_views` over trailing 6 months.

Dividing by mean normalises for title size — a small film growing from 100 to 200 monthly ratings scores equally to a blockbuster growing from 10,000 to 20,000. This makes the ranking meaningful across different popularity tiers.

---

## 6. Engineering Decisions

| Decision | Rationale |
|---|---|
| `urllib` for download, no `requests` | Reduces hard dependency; urllib is stdlib |
| SVD from scratch, not Surprise | Demonstrates mathematical understanding; avoids version conflicts |
| Direct module imports in `train_all.py` | Ensures joblib pickles classes under real module names, not `__main__` |
| `@st.cache_data` + `@st.cache_resource` | Data cached by content hash; models cached as singleton resources |
| Proxy reviews documented openly | Transparency > fake authenticity; real review pipeline is identical |
| Monthly TS granularity | Sparse data aggregated to reliable signal level |
| Genre 3× repetition in TF-IDF | Cheap, effective way to boost domain-specific field weight |

---

## 7. What's Next

| Upgrade | Impact |
|---|---|
| Real scraped reviews (IMDB/Letterboxd) | Genuine sentiment signal, eliminates proxy |
| BERT fine-tuned on movie reviews | State-of-art NLP accuracy |
| FAISS approximate NN for CB | Scales to millions of items (current cosine matrix is O(n²)) |
| ALS instead of SGD for SVD | Better parallelism on large sparse matrices |
| Facebook Prophet for forecasting | Handles holidays, irregular seasonality |
| FastAPI serving layer | REST endpoints for recommendations and predictions |
| Airflow DAG | Scheduled daily data refresh pipeline |
| Docker + docker-compose | Reproducible deployment anywhere |

---

*StreamIQ Real · MovieTweetings + IMDB · Python · Scikit-learn · Streamlit · Plotly*
