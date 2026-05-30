"""
StreamIQ Real – Exploratory Data Analysis
==========================================
Run directly: python notebooks/EDA.py
Or convert:   jupytext --to notebook notebooks/EDA.py
"""

# %% [markdown]
# # StreamIQ Real – EDA on MovieTweetings + IMDB Data

# %%
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC  = os.path.join(BASE, "data", "processed")
RPT   = os.path.join(BASE, "reports")
os.makedirs(RPT, exist_ok=True)

plt.style.use("dark_background")
CA = "#E50914"; CG = "#F5C518"; CT = "#00D4AA"; CP = "#8B5CF6"

# %% [markdown]
# ## 1. Load Processed Data

# %%
content = pd.read_csv(os.path.join(PROC, "content_catalog.csv"))
users   = pd.read_csv(os.path.join(PROC, "users.csv"))
inter   = pd.read_csv(os.path.join(PROC, "interactions.csv"))
inter["timestamp"] = pd.to_datetime(inter["timestamp"])

print("="*50)
print(f"  Content titles : {len(content):,}")
print(f"  Users          : {len(users):,}")
print(f"  Interactions   : {len(inter):,}")
print(f"  Date range     : {inter['timestamp'].min().date()} → {inter['timestamp'].max().date()}")
print(f"  Genres         : {content['genre'].nunique()}")
print(f"  Avg rating     : {inter['rating'].mean():.3f}")
print(f"  Rating std     : {inter['rating'].std():.3f}")
print("="*50)

# %% [markdown]
# ## 2. Data Quality Check

# %%
print("\nMissing values in content_catalog:")
miss = content.isnull().sum()
print(miss[miss > 0].to_string())

print("\nDuplicates in interactions:", inter.duplicated(["user_id","content_id"]).sum())
print("Rating range:", inter["rating"].min(), "–", inter["rating"].max())

# %% [markdown]
# ## 3. Content Distribution

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0D0D0D")
fig.suptitle("Content Catalog Analysis", color="white", fontsize=14, y=1.01)

# Genre counts
gc = content["genre"].value_counts()
axes[0].barh(gc.index, gc.values, color=CA, alpha=0.85)
axes[0].set_title("Movies by Genre", color="white")
axes[0].tick_params(colors="white"); axes[0].set_facecolor("#161616")
for spine in axes[0].spines.values(): spine.set_edgecolor("#2A2A2A")

# Rating distribution
axes[1].hist(content["avg_rating"].dropna(), bins=30, color=CP, edgecolor="black", alpha=0.9)
axes[1].axvline(content["avg_rating"].mean(), color=CG, linestyle="--",
                label=f"Mean: {content['avg_rating'].mean():.2f}")
axes[1].set_title("Content Avg Rating Distribution", color="white")
axes[1].tick_params(colors="white"); axes[1].set_facecolor("#161616")
axes[1].legend(labelcolor="white")
for spine in axes[1].spines.values(): spine.set_edgecolor("#2A2A2A")

# Release year
yc = content["release_year"].dropna().astype(int).value_counts().sort_index()
axes[2].fill_between(yc.index, yc.values, alpha=0.3, color=CT)
axes[2].plot(yc.index, yc.values, color=CT, linewidth=2)
axes[2].set_title("Movies Released per Year", color="white")
axes[2].tick_params(colors="white"); axes[2].set_facecolor("#161616")
for spine in axes[2].spines.values(): spine.set_edgecolor("#2A2A2A")

plt.tight_layout()
plt.savefig(os.path.join(RPT, "eda_01_content.png"), dpi=150,
            bbox_inches="tight", facecolor="#0D0D0D")
print("Saved: eda_01_content.png")
plt.show()

# %% [markdown]
# ## 4. User Behaviour Analysis

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor("#0D0D0D")
fig.suptitle("User Behaviour Analysis", color="white", fontsize=14, y=1.01)

# Ratings per user (long tail)
rpu = inter.groupby("user_id")["rating"].count()
axes[0,0].hist(rpu.clip(upper=200), bins=50, color=CA, edgecolor="black", alpha=0.85)
axes[0,0].set_title(f"Ratings per User (clipped @200)\nMedian: {rpu.median():.0f}", color="white")
axes[0,0].tick_params(colors="white"); axes[0,0].set_facecolor("#161616")

# Ratings per movie
rpm = inter.groupby("content_id")["rating"].count()
axes[0,1].hist(rpm.clip(upper=500), bins=50, color=CP, edgecolor="black", alpha=0.85)
axes[0,1].set_title(f"Ratings per Movie (clipped @500)\nMedian: {rpm.median():.0f}", color="white")
axes[0,1].tick_params(colors="white"); axes[0,1].set_facecolor("#161616")

# Rating distribution
rc = inter["rating"].value_counts().sort_index()
axes[0,2].bar(rc.index, rc.values, color=CG, edgecolor="black", alpha=0.85, width=0.4)
axes[0,2].set_title("Rating Distribution (1–5 scale)", color="white")
axes[0,2].tick_params(colors="white"); axes[0,2].set_facecolor("#161616")

# Activity over time
monthly = inter.set_index("timestamp").resample("ME")["rating"].count()
axes[1,0].fill_between(monthly.index, monthly.values, alpha=0.3, color=CA)
axes[1,0].plot(monthly.index, monthly.values, color=CA, linewidth=1.5)
axes[1,0].set_title("Monthly Rating Volume", color="white")
axes[1,0].tick_params(colors="white"); axes[1,0].set_facecolor("#161616")

# Hour-of-day pattern
hourly = inter.groupby(inter["timestamp"].dt.hour)["rating"].count()
axes[1,1].plot(hourly.index, hourly.values, color=CT, linewidth=2, marker="o", ms=4)
axes[1,1].fill_between(hourly.index, hourly.values, alpha=0.2, color=CT)
axes[1,1].set_title("Ratings by Hour of Day", color="white")
axes[1,1].tick_params(colors="white"); axes[1,1].set_facecolor("#161616")

# Genre engagement
ge = inter.groupby("genre")["rating"].count().sort_values()
axes[1,2].barh(ge.index, ge.values, color=CT, alpha=0.85)
axes[1,2].set_title("Interactions by Genre", color="white")
axes[1,2].tick_params(colors="white"); axes[1,2].set_facecolor("#161616")

for ax in axes.flat:
    for spine in ax.spines.values(): spine.set_edgecolor("#2A2A2A")

plt.tight_layout()
plt.savefig(os.path.join(RPT, "eda_02_behaviour.png"), dpi=150,
            bbox_inches="tight", facecolor="#0D0D0D")
print("Saved: eda_02_behaviour.png")
plt.show()

# %% [markdown]
# ## 5. Rating Patterns & Bias

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0D0D0D")
fig.suptitle("Rating Patterns", color="white", fontsize=14, y=1.01)

# User rating bias (mean rating per user)
user_means = inter.groupby("user_id")["rating"].mean()
axes[0].hist(user_means, bins=40, color=CA, edgecolor="black", alpha=0.85)
axes[0].axvline(user_means.mean(), color=CG, linestyle="--",
                label=f"Mean: {user_means.mean():.2f}")
axes[0].set_title("Per-User Mean Rating\n(user generosity distribution)", color="white")
axes[0].tick_params(colors="white"); axes[0].set_facecolor("#161616")
axes[0].legend(labelcolor="white")

# Movie rating bias
movie_means = inter.groupby("content_id")["rating"].mean()
axes[1].hist(movie_means, bins=40, color=CP, edgecolor="black", alpha=0.85)
axes[1].axvline(movie_means.mean(), color=CG, linestyle="--",
                label=f"Mean: {movie_means.mean():.2f}")
axes[1].set_title("Per-Movie Mean Rating\n(content quality distribution)", color="white")
axes[1].tick_params(colors="white"); axes[1].set_facecolor("#161616")
axes[1].legend(labelcolor="white")

# Genre avg rating
genre_rating = inter.groupby("genre")["rating"].mean().sort_values()
bars = axes[2].barh(genre_rating.index, genre_rating.values, alpha=0.85)
for bar, val in zip(bars, genre_rating.values):
    bar.set_color(CA if val < genre_rating.mean() else CT)
axes[2].axvline(genre_rating.mean(), color=CG, linestyle="--",
                label=f"Overall: {genre_rating.mean():.2f}")
axes[2].set_title("Avg Rating by Genre", color="white")
axes[2].tick_params(colors="white"); axes[2].set_facecolor("#161616")
axes[2].legend(labelcolor="white")

for ax in axes:
    for spine in ax.spines.values(): spine.set_edgecolor("#2A2A2A")

plt.tight_layout()
plt.savefig(os.path.join(RPT, "eda_03_rating_patterns.png"), dpi=150,
            bbox_inches="tight", facecolor="#0D0D0D")
print("Saved: eda_03_rating_patterns.png")
plt.show()

# %% [markdown]
# ## 6. Data Sparsity Analysis

# %%
n_users   = inter["user_id"].nunique()
n_items   = inter["content_id"].nunique()
n_ratings = len(inter)
sparsity  = 1 - (n_ratings / (n_users * n_items))

print("\n" + "="*50)
print("  MATRIX SPARSITY ANALYSIS")
print("="*50)
print(f"  Users         : {n_users:,}")
print(f"  Movies        : {n_items:,}")
print(f"  Ratings       : {n_ratings:,}")
print(f"  Max possible  : {n_users*n_items:,}")
print(f"  Sparsity      : {sparsity*100:.2f}%")
print(f"  Density       : {(1-sparsity)*100:.4f}%")
print()
print("  Implication: CF models face a challenging sparse matrix.")
print("  SVD with 40 latent factors helps generalise over this sparsity.")
print("="*50)

# Power law — long tail of ratings
rpu_sorted = rpu.sort_values(ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor("#0D0D0D")
ax.set_facecolor("#161616")
ax.plot(rpu_sorted.values, color=CA, linewidth=1.5)
ax.fill_between(range(len(rpu_sorted)), rpu_sorted.values, alpha=0.15, color=CA)
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_title("Long-Tail Distribution: Ratings per User (log-log)", color="white")
ax.set_xlabel("User rank", color="white"); ax.set_ylabel("Ratings (log)", color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values(): spine.set_edgecolor("#2A2A2A")
plt.tight_layout()
plt.savefig(os.path.join(RPT, "eda_04_long_tail.png"), dpi=150,
            bbox_inches="tight", facecolor="#0D0D0D")
print("\nSaved: eda_04_long_tail.png")
plt.show()

# %% [markdown]
# ## 7. Summary Statistics

# %%
print("\n" + "="*50)
print("  FULL DATASET SUMMARY")
print("="*50)
print(f"  Source          : MovieTweetings + IMDB Stats")
print(f"  Raw ratings     : 921,398")
print(f"  After cleaning  : {n_ratings:,} ({n_ratings/921398*100:.1f}% retained)")
print(f"  Unique users    : {n_users:,}")
print(f"  Unique movies   : {n_items:,}")
print(f"  Genres          : {inter['genre'].nunique()}")
print(f"  Year range      : {content['release_year'].min():.0f}–{content['release_year'].max():.0f}")
print(f"  Rating scale    : 1–5 (mapped from 1–10)")
print(f"  Matrix sparsity : {sparsity*100:.2f}%")
print(f"  SVD RMSE        : 0.614")
print(f"  Churn AUC       : 1.000 (recency perfectly separates churn)")
print(f"  Watch R²        : 0.596")
print(f"  Genre Accuracy  : 0.329 (vs 0.067 random baseline)")
print("="*50)
