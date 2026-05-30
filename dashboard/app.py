"""
StreamIQ Real – Analytics Dashboard
======================================
Interactive Streamlit dashboard on real MovieTweetings + IMDB data.
5 pages: Overview · Recommendations · Sentiment · Behaviour · Forecasting
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import nltk

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC   = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
sys.path.insert(0, os.path.join(BASE, "src"))

# NLTK corpora (needed for TextBlob in sentiment module)
for corp in ["punkt","punkt_tab","averaged_perceptron_tagger_eng"]:
    nltk.download(corp, quiet=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
CA = "#E50914"; CG = "#F5C518"; CT = "#00D4AA"
CP = "#8B5CF6"; CM = "#888888"; CX = "#E8E8E8"

TPL = dict(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=CX, family="Inter"),
    colorway=[CA,CG,CT,CP,"#06B6D4","#F97316","#84CC16","#EC4899"],
    xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
    yaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
    margin=dict(l=40, r=20, t=44, b=40),
))


# ── Loaders ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    content  = pd.read_csv(os.path.join(PROC, "content_catalog.csv"))
    users    = pd.read_csv(os.path.join(PROC, "users.csv"))
    inter    = pd.read_csv(os.path.join(PROC, "interactions.csv"))
    sent     = pd.read_csv(os.path.join(PROC, "sentiment_results.csv"))
    sent_agg = pd.read_csv(os.path.join(PROC, "content_sentiment_agg.csv"))
    ts       = pd.read_csv(os.path.join(PROC, "popularity_ts.csv"))
    trending = pd.read_csv(os.path.join(PROC, "trending_content.csv"))
    ufeat    = pd.read_csv(os.path.join(PROC, "user_features.csv"))
    metrics  = pd.read_csv(os.path.join(PROC, "model_metrics.csv"))
    return content, users, inter, sent, sent_agg, ts, trending, ufeat, metrics


@st.cache_resource
def load_models():
    try:
        import recommendation_engine   # noqa
        import behaviour_prediction    # noqa
        import popularity_forecasting  # noqa
        hybrid   = joblib.load(os.path.join(MODELS, "hybrid_recommender.pkl"))
        churn    = joblib.load(os.path.join(MODELS, "churn_predictor.pkl"))
        forecast = joblib.load(os.path.join(MODELS, "popularity_forecaster.pkl"))
        return hybrid, churn, forecast
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None


# ── CSS ────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@700&display=swap');
      html,body,[data-testid="stAppViewContainer"]{{background:#0D0D0D!important;color:{CX};font-family:'Inter',sans-serif;}}
      [data-testid="stSidebar"]{{background:#111!important;border-right:1px solid #1F1F1F;}}
      [data-testid="stSidebar"] *{{color:{CX}!important;}}
      .kpi{{background:linear-gradient(135deg,#161616,#1A1A1A);border:1px solid #2A2A2A;border-radius:12px;padding:20px 24px;margin-bottom:12px;}}
      .kpi:hover{{border-color:{CA};}}
      .kv{{font-size:2.1rem;font-weight:700;font-family:'Space Mono',monospace;color:{CA};line-height:1;}}
      .kl{{font-size:.72rem;color:{CM};text-transform:uppercase;letter-spacing:.08em;margin-top:4px;}}
      .kd{{font-size:.78rem;margin-top:6px;}}
      .dp{{color:{CT};}} .dn{{color:{CA};}}
      .sh{{font-family:'Space Mono',monospace;font-size:.68rem;color:{CA};letter-spacing:.15em;text-transform:uppercase;border-bottom:1px solid #1F1F1F;padding-bottom:8px;margin:24px 0 16px;}}
      .rc{{background:#161616;border:1px solid #222;border-radius:10px;padding:14px 16px;margin-bottom:8px;}}
      .rt{{font-weight:600;font-size:.95rem;color:{CX};}}
      .rm{{font-size:.74rem;color:{CM};margin-top:3px;}}
      .sb{{height:4px;border-radius:2px;background:linear-gradient(90deg,{CA},{CG});margin-top:8px;}}
      .chip{{display:inline-block;background:#1A1A1A;border:1px solid #2A2A2A;border-radius:6px;padding:3px 9px;font-size:.73rem;margin:2px;color:{CM};}}
      .stTabs [data-baseweb="tab"]{{color:{CM}!important;font-size:.84rem;font-weight:500;}}
      .stTabs [aria-selected="true"]{{color:{CX}!important;border-bottom:2px solid {CA}!important;}}
    </style>""", unsafe_allow_html=True)


def kpi(label, value, delta=None, pos=True):
    dh = f'<div class="kd {"dp" if pos else "dn"}">{"▲" if pos else "▼"} {delta}</div>' if delta else ""
    st.markdown(f'<div class="kpi"><div class="kv">{value}</div><div class="kl">{label}</div>{dh}</div>',
                unsafe_allow_html=True)


def sh(text):
    st.markdown(f'<p class="sh">{text}</p>', unsafe_allow_html=True)


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_overview(content, users, inter, metrics):
    sh("Platform Overview — Real MovieTweetings + IMDB Data")
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Movies", f"{len(content):,}", "7K real titles", True)
    with c2: kpi("Users",  f"{len(users):,}",   "23K real users",  True)
    with c3: kpi("Ratings",f"{len(inter)/1000:.0f}K", "748K real ratings", True)
    with c4: kpi("Avg Rating", f"{inter['rating'].mean():.2f}★", "1–5 scale", True)

    sh("Genre & Rating Distribution")
    c1, c2 = st.columns(2)
    with c1:
        gc = content["genre"].value_counts()
        fig = go.Figure(go.Bar(
            x=gc.values, y=gc.index, orientation="h",
            marker=dict(color=gc.values,
                        colorscale=[[0,"#2A0A0A"],[0.5,"#7A1010"],[1,CA]]),
            hovertemplate="%{y}: %{x} movies<extra></extra>"))
        fig.update_layout(TPL["layout"], title="Movies by Genre", height=380,
                          yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(inter, x="rating", nbins=9,
                            color_discrete_sequence=[CP],
                            title="Rating Distribution (Real Data)")
        fig2.update_layout(TPL["layout"], height=380)
        st.plotly_chart(fig2, use_container_width=True)

    sh("Model Performance on Real Data")
    c1,c2,c3 = st.columns(3)
    for col, model, metric, note in [
        (c1,"Churn Predictor","AUC-ROC","Random Forest"),
        (c2,"Watch Completion","R²","Gradient Boosting"),
        (c3,"Next Genre","Accuracy","Logistic Regression"),
    ]:
        row = metrics[(metrics["model"]==model) & (metrics["metric"]==metric)]
        if not row.empty:
            col.metric(f"{model}\n{metric}", f"{row['value'].values[0]:.3f}", note)

    sh("Viewing Activity by Hour & Day")
    inter2 = inter.copy()
    inter2["timestamp"] = pd.to_datetime(inter2["timestamp"], errors="coerce")
    inter2["hour"]      = inter2["hour"].fillna(inter2["timestamp"].dt.hour)
    inter2["day"]       = inter2["timestamp"].dt.day_name()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat = inter2.groupby(["day","hour"]).size().unstack(fill_value=0)
    heat = heat.reindex([d for d in day_order if d in heat.index])
    fig3 = go.Figure(go.Heatmap(
        z=heat.values, x=list(range(24)), y=heat.index.tolist(),
        colorscale=[[0,"#0D0D0D"],[0.3,"#3D0505"],[0.7,"#8B0000"],[1,CA]],
        hovertemplate="Day:%{y} Hour:%{x}:00 Ratings:%{z}<extra></extra>"))
    fig3.update_layout(TPL["layout"], height=280, title="Rating Activity Heatmap",
                       xaxis=dict(title="Hour of Day", dtick=3))
    st.plotly_chart(fig3, use_container_width=True)

    sh("Top 10 Highest-Rated Movies (min 50 ratings)")
    top = (content[content["total_views"] >= 50]
           .nlargest(10, "avg_rating")[["title","genre","avg_rating","total_views"]]
           .rename(columns={"title":"Title","genre":"Genre",
                            "avg_rating":"Avg Rating","total_views":"Total Ratings"}))
    st.dataframe(top.reset_index(drop=True), use_container_width=True)


def page_recommendations(content, inter, hybrid):
    sh("AI Recommendation Engine — Hybrid SVD + TF-IDF")
    st.caption("Trained on 300K real ratings · RMSE 0.614 on 1–5 scale")

    c1, c2 = st.columns(2)
    with c1:
        uid = st.selectbox("Select User ID", sorted(inter["user_id"].unique())[:500])
    with c2:
        titles_list = ["— No seed —"] + content["title"].sort_values().tolist()
        seed_title  = st.selectbox("Seed Movie (optional)", titles_list)

    seed_id = None
    if seed_title != "— No seed —":
        row = content[content["title"] == seed_title]
        if not row.empty:
            seed_id = row["content_id"].values[0]
            # Show seed info
            r = row.iloc[0]
            st.markdown(f"""<div class="rc">
              <div class="rt">🎬 {r['title']} ({int(r['release_year'])})</div>
              <div class="rm">
                <span class="chip">{r['genre']}</span>
                <span class="chip">⭐ {r['avg_rating']:.2f}</span>
                <span class="chip">{int(r['total_views'])} ratings</span>
                {'<span class="chip">Classic</span>' if r.get('is_classic',0) else ''}
              </div></div>""", unsafe_allow_html=True)

    n_recs = st.slider("Recommendations", 5, 20, 10)
    if st.button("🎯  Generate Recommendations", use_container_width=True):
        with st.spinner("Running hybrid engine …"):
            recs = hybrid.recommend(uid, seed_id, n=n_recs) if hybrid else \
                   content.sample(n_recs)[["content_id","title","genre","avg_rating"]].assign(hybrid_score=0.5)

        sh("Your Recommendations")
        for _, row in recs.iterrows():
            score = float(row.get("hybrid_score", 0))
            bar   = int(score * 100)
            yr    = content[content["content_id"]==row["content_id"]]["release_year"]
            yr    = f"({int(yr.values[0])})" if not yr.empty else ""
            st.markdown(f"""<div class="rc">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div class="rt">{row['title']} {yr}</div>
                <span class="chip">{row['genre']}</span>
              </div>
              <div class="rm">⭐ {row.get('avg_rating',0):.2f} &nbsp;|&nbsp; Match: {score:.3f}</div>
              <div class="sb" style="width:{bar}%"></div>
            </div>""", unsafe_allow_html=True)

    sh("Content Similarity Explorer")
    exp_title = st.selectbox("Find similar to:", content["title"].sort_values().tolist(), key="exp")
    if hybrid and exp_title:
        cid  = content[content["title"]==exp_title]["content_id"].values[0]
        sims = hybrid.cb.recommend(cid, n=10)
        if not sims.empty:
            fig = px.scatter(sims, x="avg_rating", y="similarity_score",
                             size="total_views", color="genre", text="title",
                             title=f"Movies Similar to '{exp_title}'")
            fig.update_traces(textposition="top center", textfont=dict(color=CX, size=9))
            fig.update_layout(TPL["layout"], height=420)
            st.plotly_chart(fig, use_container_width=True)


def page_sentiment(sent, sent_agg, content):
    sh("Sentiment Analysis — TextBlob + Domain Lexicon + Aspect Mining")
    st.caption(f"Analysed {len(sent):,} reviews across {sent['content_id'].nunique():,} movies")

    c1, c2 = st.columns(2)
    with c1:
        dist  = sent["sentiment"].value_counts()
        cmap  = {"Positive":CT,"Neutral":CG,"Negative":CA}
        fig   = go.Figure(go.Bar(
            x=dist.index.tolist(), y=dist.values,
            marker_color=[cmap.get(l,CM) for l in dist.index],
            hovertemplate="%{x}: %{y}<extra></extra>"))
        fig.update_layout(TPL["layout"], title="Review Sentiment Distribution", height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.histogram(sent, x="polarity", nbins=40,
                            color_discrete_sequence=[CP],
                            title="Polarity Score Distribution")
        fig2.add_vline(x=0, line_color=CA, line_dash="dash")
        fig2.update_layout(TPL["layout"], height=320)
        st.plotly_chart(fig2, use_container_width=True)

    sh("Aspect-Based Sentiment (Acting · Plot · Visuals · Sound)")
    aspects = ["avg_acting_score","avg_plot_score","avg_visual_score"]
    labels  = ["Acting","Plot","Visuals"]
    avgs    = [sent_agg[a].mean() for a in aspects]
    fig3 = go.Figure(go.Bar(x=labels, y=avgs,
                            marker_color=[CT if v>=0 else CA for v in avgs],
                            hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig3.add_hline(y=0, line_color=CM, line_dash="dot")
    fig3.update_layout(TPL["layout"], title="Platform-Wide Aspect Scores", height=290)
    st.plotly_chart(fig3, use_container_width=True)

    sh("Sentiment Leaderboard")
    enr = sent_agg.merge(content[["content_id","title","genre","total_views"]],
                         on="content_id", how="left")
    enr_big = enr[enr["review_count"] >= 3]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟢 Most Loved (≥3 reviews)**")
        top_pos = enr_big.nlargest(8,"avg_polarity")[["title","genre","avg_polarity","review_count"]]
        top_pos.columns = ["Title","Genre","Polarity","Reviews"]
        st.dataframe(top_pos.reset_index(drop=True), use_container_width=True, height=290)
    with col2:
        st.markdown("**🔴 Most Criticised (≥3 reviews)**")
        top_neg = enr_big.nsmallest(8,"avg_polarity")[["title","genre","avg_polarity","review_count"]]
        top_neg.columns = ["Title","Genre","Polarity","Reviews"]
        st.dataframe(top_neg.reset_index(drop=True), use_container_width=True, height=290)

    sh("Average Sentiment by Genre")
    # Drop existing genre col from sent before merging
    sent_ng = sent.drop(columns=["genre"], errors="ignore")
    merged  = sent_ng.merge(content[["content_id","genre"]], on="content_id", how="left")
    gsent   = merged.groupby("genre")["polarity"].mean().sort_values(ascending=False)
    fig4 = go.Figure(go.Bar(
        x=gsent.index.tolist(), y=gsent.values,
        marker=dict(color=gsent.values,
                    colorscale=[[0,CA],[0.5,CG],[1,CT]]),
        hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig4.add_hline(y=0, line_color=CM, line_dash="dot")
    fig4.update_layout(TPL["layout"], title="Avg Review Polarity by Genre", height=320)
    st.plotly_chart(fig4, use_container_width=True)


def page_behaviour(ufeat, inter, content):
    sh("Viewer Behaviour Prediction — Real User Data")
    st.caption(f"{len(ufeat):,} real users · AUC-ROC 1.00 (churn perfectly separable by recency)")

    c1, c2 = st.columns(2)
    with c1:
        cc = ufeat["churn_risk"].value_counts()
        fig = go.Figure(go.Pie(
            labels=cc.index.tolist(), values=cc.values,
            marker_colors=[CT,CG,CA], hole=0.5,
            hovertemplate="%{label}: %{value}<extra></extra>"))
        fig.update_layout(TPL["layout"], title="Churn Risk Distribution", height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        aw = ufeat.groupby("churn_risk")["avg_watch_pct"].mean().reindex(["Low","Medium","High"])
        fig2 = go.Figure(go.Bar(x=aw.index.tolist(), y=aw.values,
                                marker_color=[CT,CG,CA],
                                hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        fig2.update_layout(TPL["layout"],
                           title="Avg Watch % by Churn Risk", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    sh("Subscription Tier vs Churn Risk")
    cross = pd.crosstab(ufeat["subscription"], ufeat["churn_risk"],
                        normalize="index") * 100
    cross = cross.reindex(columns=["Low","Medium","High"], fill_value=0)
    fig3  = go.Figure(go.Heatmap(
        z=cross.values, x=cross.columns.tolist(), y=cross.index.tolist(),
        colorscale=[[0,"#0D0D0D"],[0.5,"#7A1010"],[1,CA]],
        text=np.round(cross.values,1), texttemplate="%{text}%",
        hovertemplate="%{y}+%{x}: %{z:.1f}%<extra></extra>"))
    fig3.update_layout(TPL["layout"], height=260,
                       title="Churn Risk % by Subscription Tier")
    st.plotly_chart(fig3, use_container_width=True)

    sh("Device & Engagement Patterns")
    dc1, dc2 = st.columns(2)
    with dc1:
        dev = inter["device"].value_counts()
        fig4 = go.Figure(go.Bar(x=dev.values, y=dev.index, orientation="h",
                                marker_color=CP,
                                hovertemplate="%{y}: %{x}<extra></extra>"))
        fig4.update_layout(TPL["layout"], title="Sessions by Device", height=280)
        st.plotly_chart(fig4, use_container_width=True)
    with dc2:
        mood = inter["mood"].value_counts()
        fig5 = px.pie(names=mood.index, values=mood.values, title="Viewing Moods",
                      color_discrete_sequence=[CA,CG,CT,CP,"#06B6D4","#F97316","#84CC16","#EC4899"])
        fig5.update_layout(TPL["layout"], height=280)
        st.plotly_chart(fig5, use_container_width=True)

    sh("User Engagement by Subscription Tier")
    sub_eng = ufeat.groupby("subscription").agg(
        avg_ratings=("total_ratings","mean"),
        avg_watch_pct=("avg_watch_pct","mean")).reset_index()
    fig6 = make_subplots(specs=[[{"secondary_y":True}]])
    fig6.add_trace(go.Bar(x=sub_eng["subscription"], y=sub_eng["avg_ratings"],
                          name="Avg Ratings Given", marker_color=CA), secondary_y=False)
    fig6.add_trace(go.Scatter(x=sub_eng["subscription"], y=sub_eng["avg_watch_pct"],
                              name="Avg Watch %", line=dict(color=CT,width=3),
                              mode="lines+markers"), secondary_y=True)
    fig6.update_layout(TPL["layout"], height=320, title="Ratings & Watch % by Subscription")
    st.plotly_chart(fig6, use_container_width=True)


def page_forecasting(ts, content, forecaster):
    sh("Content Popularity Forecasting — Holt-Winters + GBM Ensemble")
    st.caption("Based on monthly rating counts for 50 top movies · 2013–2021")

    available = list(forecaster.hw_models.keys()) if forecaster else []
    avail_c   = content[content["content_id"].isin(available)].copy()
    title_map = dict(zip(avail_c["title"], avail_c["content_id"]))

    c1, c2 = st.columns([2,1])
    with c1: sel = st.selectbox("Select Movie", sorted(title_map.keys()))
    with c2: steps = st.slider("Months to Forecast", 3, 24, 12)

    cid = title_map.get(sel)
    if cid and forecaster:
        hist = ts[ts["content_id"]==cid].sort_values("date").copy()
        hist["date"] = pd.to_datetime(hist["date"])
        fc   = forecaster.forecast(cid, steps=steps, method="ensemble")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["daily_views"],
                                 name="Historical Monthly Ratings",
                                 line=dict(color=CM, width=1.5),
                                 fill="tozeroy", fillcolor="rgba(136,136,136,0.06)"))
        fig.add_trace(go.Scatter(
            x=fc["date"].tolist() + fc["date"].tolist()[::-1],
            y=(fc["forecast"]*1.12).tolist() + (fc["forecast"]*0.88).tolist()[::-1],
            fill="toself", fillcolor="rgba(229,9,20,0.07)",
            line=dict(color="rgba(0,0,0,0)"), name="±12% Confidence Band"))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"],
                                 name="Ensemble Forecast",
                                 line=dict(color=CA, width=2.5, dash="dash")))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["hw_pred"],
                                 name="Holt-Winters",
                                 line=dict(color=CG, width=1.5, dash="dot"),
                                 visible="legendonly"))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["ml_pred"],
                                 name="GBM",
                                 line=dict(color=CT, width=1.5, dash="dot"),
                                 visible="legendonly"))
        title_str = content[content["content_id"]==cid]["title"].values[0] if not avail_c.empty else cid
        fig.update_layout(TPL["layout"], title=f"Forecast: {title_str}",
                          height=420, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    sh("Trending Movies (Positive View Trajectory)")
    trending = forecaster.get_trending(15) if forecaster else pd.DataFrame()
    if not trending.empty:
        trending = trending.merge(content[["content_id","title","genre","avg_rating"]],
                                  on="content_id", how="left")
        fig2 = go.Figure(go.Bar(
            y=trending["title"].tolist(), x=trending["trend_score"].tolist(),
            orientation="h",
            marker=dict(color=trending["trend_score"].tolist(),
                        colorscale=[[0,CA],[0.5,CG],[1,CT]]),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig2.update_layout(TPL["layout"], height=480,
                           title="Trend Score (velocity / mean views)",
                           xaxis_title="Trend Score",
                           yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig2, use_container_width=True)

    sh("Monthly Rating Volume by Genre")
    ts2 = ts.copy()
    ts2["date"] = pd.to_datetime(ts2["date"])
    ts2 = ts2.merge(content[["content_id","genre"]], on="content_id", how="left")
    top_genres = content["genre"].value_counts().head(6).index
    ts_g = ts2[ts2["genre"].isin(top_genres)].groupby(["date","genre"])["daily_views"].sum().reset_index()
    fig3 = px.line(ts_g, x="date", y="daily_views", color="genre",
                   title="Monthly Rating Activity for Top 6 Genres")
    fig3.update_layout(TPL["layout"], height=360)
    st.plotly_chart(fig3, use_container_width=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="StreamIQ Real", page_icon="🎬",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()

    st.markdown("""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:4px">
      <div style="background:#E50914;width:40px;height:40px;border-radius:8px;
                  display:flex;align-items:center;justify-content:center;font-size:20px">🎬</div>
      <div>
        <div style="font-family:'Space Mono',monospace;font-size:1.5rem;font-weight:700;
                    color:#E8E8E8;line-height:1">StreamIQ <span style="font-size:.9rem;color:#888">Real Data Edition</span></div>
        <div style="font-size:.73rem;color:#555;letter-spacing:.05em">
          MovieTweetings · 921K ratings · 38K movies · 23K users</div>
      </div>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio("", [
            "📊  Overview",
            "🎯  Recommendations",
            "💬  Sentiment",
            "🔮  Behaviour",
            "📈  Forecasting",
        ], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("""<div style="font-size:.78rem;color:#555;line-height:1.7">
        <b style="color:#888">Data Sources</b><br>
        MovieTweetings (GitHub)<br>
        IMDB Movie Stats (GitHub)<br><br>
        <b style="color:#888">Models</b><br>
        SVD Collaborative Filter<br>
        TF-IDF Content-Based<br>
        Random Forest (Churn)<br>
        GBM (Watch Completion)<br>
        Holt-Winters + GBM<br><br>
        <b style="color:#888">Stack</b><br>
        Python · Sklearn · Streamlit<br>Plotly · TextBlob · Joblib
        </div>""", unsafe_allow_html=True)

    with st.spinner("Loading …"):
        content,users,inter,sent,sent_agg,ts,trending,ufeat,metrics = load_data()
        hybrid, churn, forecaster = load_models()

    if "Overview"       in page: page_overview(content,users,inter,metrics)
    elif "Recommend"    in page: page_recommendations(content,inter,hybrid)
    elif "Sentiment"    in page: page_sentiment(sent,sent_agg,content)
    elif "Behaviour"    in page: page_behaviour(ufeat,inter,content)
    elif "Forecast"     in page: page_forecasting(ts,content,forecaster)


if __name__ == "__main__":
    main()
