"""
StreamIQ Real – Sentiment Analysis
====================================
Since MovieTweetings has no text reviews, we:
  1. Generate sentiment-rich review proxies from rating + genre + year
     (simulates what a real pipeline would do with scraped reviews)
  2. Apply TextBlob + domain lexicon + aspect mining  (same as before)
  3. Aggregate per-content
"""

import os, re, random, warnings
import pandas as pd
import numpy as np
from textblob import TextBlob
import joblib
warnings.filterwarnings("ignore")

PROC   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS, exist_ok=True)

random.seed(42); np.random.seed(42)

# ── Domain lexicon ─────────────────────────────────────────────────────────────
POS_LEX = {
    "masterpiece":2.0,"brilliant":1.8,"outstanding":1.8,"gripping":1.5,
    "captivating":1.7,"binge":1.5,"breathtaking":2.0,"stellar":1.6,
    "phenomenal":2.0,"riveting":1.7,"must watch":1.8,"flawless":2.0,
    "unforgettable":1.9,"amazing":1.5,"fantastic":1.5,"superb":1.7,
    "incredible":1.6,"loved":1.3,"perfect":1.8,"classic":1.4,
    "timeless":1.6,"iconic":1.5,"cinematic":1.3,"moving":1.4,
}
NEG_LEX = {
    "boring":-1.5,"terrible":-2.0,"awful":-2.0,"dreadful":-2.0,
    "predictable":-1.2,"clichéd":-1.3,"disappointing":-1.7,
    "overrated":-1.5,"unwatchable":-2.0,"dull":-1.4,"bland":-1.3,
    "forgettable":-1.3,"poor":-1.2,"mediocre":-1.0,"slow":-0.8,
    "waste":-1.6,"skip":-1.2,"pretentious":-1.3,"incoherent":-1.4,
}
ASPECT_KW = {
    "acting":  ["acting","actor","actress","performance","cast","portrayal","role"],
    "plot":    ["plot","story","script","writing","narrative","twist","ending","pacing"],
    "visuals": ["visuals","cinematography","cgi","effects","animation","stunning","photography"],
    "sound":   ["music","soundtrack","score","audio","sound","songs","theme"],
}

# ── Review templates per rating bucket ────────────────────────────────────────

TEMPLATES = {
    "high": [
        "An absolute {a}. The {b} is {c} and the acting is stellar.",
        "One of the most {a} films I have ever seen. {c} storytelling.",
        "A true {a}. Breathtaking visuals and a gripping narrative.",
        "Completely {c}. The plot is brilliant and the cast is outstanding.",
        "A cinematic {a}. Incredible performances and stunning cinematography.",
        "This is a timeless {a}. The soundtrack is phenomenal and the script is flawless.",
    ],
    "mid": [
        "A decent {b} with some {c} moments. Not without flaws but worth watching.",
        "Passable {b}. The acting is okay but the plot feels a bit slow.",
        "Has its moments. Some scenes are {c} but overall feels mediocre.",
        "Average {b}. Nothing groundbreaking, but not boring either.",
        "Mixed feelings. Strong performances let down by a predictable story.",
    ],
    "low": [
        "Disappointing {b}. The plot is boring and the acting is poor.",
        "A dull and forgettable {b}. Terrible pacing and bland characters.",
        "Awful. Waste of time — predictable, clichéd, and unwatchable.",
        "Overrated and dreadful. The script is incoherent and the ending is awful.",
        "One of the worst {b}s in recent memory. Skip this mediocre film.",
    ],
}

ADJ_HIGH  = ["masterpiece","gem","classic","triumph","revelation"]
ADJ_MID   = ["film","movie","watch","experience","effort"]
ADJ_QUAL  = ["captivating","moving","powerful","immersive","thought-provoking"]


def make_review(rating: float, genre: str) -> str:
    g = genre.lower() if isinstance(genre, str) else "film"
    if rating >= 4.0:
        t = random.choice(TEMPLATES["high"])
        return t.format(a=random.choice(ADJ_HIGH), b=g, c=random.choice(ADJ_QUAL))
    elif rating >= 3.0:
        t = random.choice(TEMPLATES["mid"])
        return t.format(b=g, c=random.choice(["decent","interesting","solid"]))
    else:
        t = random.choice(TEMPLATES["low"])
        return t.format(b=g)


# ── Core analysis ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lexicon_boost(text: str) -> float:
    score = 0.0
    for w, v in POS_LEX.items():
        if w in text: score += v
    for w, v in NEG_LEX.items():
        if w in text: score += v
    return float(np.clip(score / 5.0, -1.0, 1.0))


def analyse_one(text: str) -> dict:
    clean  = clean_text(text)
    blob   = TextBlob(clean)
    tb_pol = blob.sentiment.polarity
    tb_sub = blob.sentiment.subjectivity
    boost  = lexicon_boost(clean)
    combined = float(np.clip(0.6 * tb_pol + 0.4 * boost, -1.0, 1.0))

    label = ("Positive" if combined >= 0.15
             else "Negative" if combined <= -0.15
             else "Neutral")

    aspects = {}
    for asp, kws in ASPECT_KW.items():
        sents = [s.raw for s in blob.sentences if any(k in s.raw.lower() for k in kws)]
        aspects[asp] = float(np.mean([TextBlob(s).sentiment.polarity for s in sents])) if sents else 0.0

    return {
        "polarity":      round(combined, 4),
        "subjectivity":  round(tb_sub, 4),
        "sentiment":     label,
        "aspect_acting": round(aspects["acting"], 3),
        "aspect_plot":   round(aspects["plot"], 3),
        "aspect_visuals":round(aspects["visuals"], 3),
        "aspect_sound":  round(aspects["sound"], 3),
    }


def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("content_id")
    return g.agg(
        avg_polarity     = ("polarity",       "mean"),
        review_count     = ("polarity",       "count"),
        pct_positive     = ("sentiment",      lambda x: (x=="Positive").mean()*100),
        pct_neutral      = ("sentiment",      lambda x: (x=="Neutral").mean()*100),
        pct_negative     = ("sentiment",      lambda x: (x=="Negative").mean()*100),
        avg_subjectivity = ("subjectivity",   "mean"),
        avg_acting_score = ("aspect_acting",  "mean"),
        avg_plot_score   = ("aspect_plot",    "mean"),
        avg_visual_score = ("aspect_visuals", "mean"),
    ).round(3).reset_index()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("="*55)
    print("  STREAMIQ REAL – SENTIMENT ANALYSIS")
    print("="*55)

    interactions = pd.read_csv(os.path.join(PROC, "interactions.csv"))
    content      = pd.read_csv(os.path.join(PROC, "content_catalog.csv"))

    genre_map = dict(zip(content["content_id"], content["genre"]))

    # Build review corpus: sample up to 5 reviews per movie from raters
    print("\n  Building review corpus from ratings …")
    rated = interactions.dropna(subset=["rating"]).copy()
    # Sample up to 5 rows per content_id without losing the column
    rng = __import__('numpy').random.default_rng(42)
    sampled = (rated.assign(_rnd=rng.random(len(rated)))
                    .sort_values(["content_id","_rnd"])
                    .groupby("content_id").head(5)
                    .drop(columns=["_rnd"])
                    .reset_index(drop=True))

    sampled["genre_label"] = sampled["content_id"].map(genre_map)
    sampled["review"] = sampled.apply(
        lambda r: make_review(r["rating"], r["genre_label"]), axis=1)

    print(f"  Analysing {len(sampled):,} reviews …")
    results = sampled["review"].apply(analyse_one).apply(pd.Series)
    sampled = pd.concat([sampled.reset_index(drop=True), results], axis=1)

    # Save full results
    sampled.to_csv(os.path.join(PROC, "sentiment_results.csv"), index=False)

    # Aggregate per content
    agg = aggregate_sentiment(sampled)
    agg.to_csv(os.path.join(PROC, "content_sentiment_agg.csv"), index=False)

    # Save lexicon
    joblib.dump({"positive": POS_LEX, "negative": NEG_LEX},
                os.path.join(MODELS, "sentiment_lexicon.pkl"))

    dist = sampled["sentiment"].value_counts(normalize=True) * 100
    print("\n  Sentiment Distribution:")
    for label, pct in dist.items():
        print(f"    {label:10s}: {pct:.1f}%")
    print(f"\n  ✅  {len(agg):,} content records aggregated.")


if __name__ == "__main__":
    main()
