"""
StreamIQ Real – Master Training Pipeline
Imports each module directly so joblib pickles classes under their
real module names (not __main__), fixing the unpickling error in dashboard.
"""

import os, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "data"))


def run_step(name, fn):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    fn()
    print(f"  ⏱  {name} done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import ingest_and_clean
    import sentiment_analysis
    import behaviour_prediction
    import popularity_forecasting
    import recommendation_engine

    print("\n" + "🎬"*30)
    print("  STREAMIQ REAL – FULL PIPELINE")
    print("🎬"*30)

    t0 = time.time()
    run_step("Data Ingestion & Cleaning", ingest_and_clean.main)
    run_step("Sentiment Analysis",        sentiment_analysis.main)
    run_step("Behaviour Prediction",      behaviour_prediction.main)
    run_step("Popularity Forecasting",    popularity_forecasting.main)
    run_step("Recommendation Engine",     recommendation_engine.main)

    print(f"\n\n{'='*60}")
    print(f"  ✅  ALL DONE in {time.time()-t0:.0f}s")
    print(f"{'='*60}\n")
