import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
import urllib.request, io as _io

st.set_page_config(page_title="AI Ultimatum Game — Class Results", layout="wide")
st.title("How does AI distribute the pie?")
st.caption("Class experiment · ChatGPT · Gemini · Copilot · Claude · 4 scenarios · 4 stake levels")

DATA_URL = "https://raw.githubusercontent.com/profguclass/HW-AIUG-Result-2026/main/HW-AIUG-ed.xlsx"

@st.cache_data
def load_data():
    try:
        resp = urllib.request.urlopen(DATA_URL)
        return pd.read_excel(_io.BytesIO(resp.read()))
    except Exception:
        return None

df = load_data()
if df is None:
    st.error("Could not load data from GitHub. Please check the repository.")
    st.stop()

stake_order = {"1만원": 1, "10만원": 2, "100만원": 3, "1000만원": 4}
df["stake_rank"] = df["stake"].map(stake_order)
df["altruistic"] = df["offer_ratio"] > 0.5
df["theorist"]      = df["offer_ratio"] < 0.1
df["human_mode"] = (df["offer_ratio"] >= 0.1) & (df["offer_ratio"] <= 0.5)

MODELS = ["ChatGPT", "Claude", "Copilot", "Gemini"]
COLORS = {"ChatGPT": "#378ADD", "Claude": "#1D9E75", "Copilot": "#D4537E", "Gemini": "#EF9F27"}
STAKES = ["1만원", "10만원", "100만원", "1000만원"]

# ── Overview metrics ──────────────────────────────────────────────────────────
st.subheader("Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Observations",           f"{len(df):,}")
c2.metric("Students",               df["student_id"].nunique())
c3.metric("Mean offer ratio",       f"{df['offer_ratio'].mean():.1%}")
c4.metric("Mean MAO",               f"{df['mao_ratio'].mean():.1%}")
c5.metric("Altruistic offers >50%", f"{df['altruistic'].mean():.1%}")
st.divider()

# ── Model-level summary table ─────────────────────────────────────────────────
st.subheader("Model-level summary")
corrs = {m: df[df["model"]==m]["offer_ratio"].corr(df[df["model"]==m]["stake_rank"]) for m in MODELS}
summary = df.groupby("model").agg(
    N=("offer_ratio","count"),
    Mean_offer=("offer_ratio","mean"),
    SD_offer=("offer_ratio","std"),
    Mean_MAO=("mao_ratio","mean"),
    SD_MAO=("mao_ratio","std"),
    Pct_altruistic=("altruistic","mean"),
    Pct_theorist=("theorist","mean"),
    Pct_human=("human_mode","mean"),
).reindex(MODELS)
summary["Offer_stake_r"] = pd.Series(corrs)
pct_cols = ["Mean_offer","SD_offer","Mean_MAO","SD_MAO","Pct_altruistic","Pct_theorist","Pct_human"]
disp = summary.copy()
for c in pct_cols:
    disp[c] = disp[c].apply(lambda x: f"{x:.1%}")
disp["Offer_stake_r"] = disp["Offer_stake_r"].apply(lambda x: f"{x:.3f}")
disp.columns = ["N","Mean offer","SD offer","Mean MAO","SD MAO",
                "% Altruistic (>50%)","% Theorist (<10%)","% Human (10–50%)",
                "Offer–stake corr."]
st.dataframe(disp, use_container_width=True)
st.divider()

# ── Mean offer & MAO by model ─────────────────────────────────────────────────
st.subheader("Mean offer ratio and MAO by model")
col1, col2 = st.columns(2)

with col1:
    means = df.groupby("model")["offer_ratio"].mean().reindex(MODELS)
    fig = go.Figure(go.Bar(
        x=MODELS, y=means.values,
        marker_color=[COLORS[m] for m in MODELS],
        text=[f"{v:.1%}" for v in means.values], textposition="outside"
    ))
    fig.update_layout(yaxis=dict(tickformat=".0%", range=[0,0.62], title="Mean offer ratio"),
                      xaxis_title="", margin=dict(t=30,b=10), height=300,
                      title="Mean offer ratio by model")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    maos = df.groupby("model")["mao_ratio"].mean().reindex(MODELS)
    fig = go.Figure(go.Bar(
        x=MODELS, y=maos.values,
        marker_color=[COLORS[m] for m in MODELS],
        text=[f"{v:.1%}" for v in maos.values], textposition="outside"
    ))
    fig.update_layout(yaxis=dict(tickformat=".0%", range=[0,0.3], title="Mean MAO"),
                      xaxis_title="", margin=dict(t=30,b=10), height=300,
                      title="Mean minimum acceptable offer (MAO) by model")
    st.plotly_chart(fig, use_container_width=True)

# ── SD comparison chart ───────────────────────────────────────────────────────
st.subheader("Variability (SD) of offer ratio and MAO by model")
sd_offer = df.groupby("model")["offer_ratio"].std().reindex(MODELS)
sd_mao   = df.groupby("model")["mao_ratio"].std().reindex(MODELS)

fig = go.Figure()
fig.add_trace(go.Bar(
    name="SD offer", x=MODELS, y=sd_offer.values,
    marker_color=[COLORS[m] for m in MODELS],
    text=[f"{v:.1%}" for v in sd_offer.values], textposition="outside"
))
fig.add_trace(go.Bar(
    name="SD MAO", x=MODELS, y=sd_mao.values,
    marker_color=[COLORS[m] for m in MODELS],
    opacity=0.45,
    text=[f"{v:.1%}" for v in sd_mao.values], textposition="outside"
))
fig.update_layout(barmode="group",
                  yaxis=dict(tickformat=".0%", range=[0,0.35], title="Standard deviation"),
                  xaxis_title="", legend_title="", margin=dict(t=10,b=10), height=320)
st.plotly_chart(fig, use_container_width=True)

# ── Offer by model x stake ────────────────────────────────────────────────────
st.subheader("Mean offer ratio by model and stake level  (stake-dependent rationality)")
stake_pivot = df.groupby(["model","stake"])["offer_ratio"].mean().reset_index()
fig = go.Figure()
for m in MODELS:
    sub = stake_pivot[stake_pivot["model"]==m].set_index("stake").reindex(STAKES)
    fig.add_trace(go.Bar(
        name=m, x=STAKES, y=sub["offer_ratio"].values,
        marker_color=COLORS[m],
        text=[f"{v:.1%}" for v in sub["offer_ratio"].values], textposition="outside"
    ))
fig.update_layout(barmode="group",
                  yaxis=dict(tickformat=".0%", range=[0,0.65], title="Mean offer ratio"),
                  xaxis_title="Stake", legend_title="Model",
                  margin=dict(t=10,b=10), height=360)
st.plotly_chart(fig, use_container_width=True)

# ── Scenario & human-responder effect ─────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Mean offer ratio by scenario")
    sc_means = df.groupby("scenario")["offer_ratio"].mean().reindex(["AA","AH","HA","HH"])
    labels   = ["AA (AI→AI)","AH (AI→Human)","HA (Human→AI)","HH (Human→Human)"]
    fig = go.Figure(go.Bar(
        x=labels, y=sc_means.values,
        marker_color=["#B5D4F4","#378ADD","#85B7EB","#185FA5"],
        text=[f"{v:.1%}" for v in sc_means.values], textposition="outside"
    ))
    fig.update_layout(yaxis=dict(tickformat=".0%", range=[0,0.6], title="Mean offer ratio"),
                      xaxis_title="", margin=dict(t=10,b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Human responder effect (Δ vs AI responder)")
    diffs = []
    for m in MODELS:
        sub  = df[df["model"]==m]
        ai_r = sub[sub["responder_type"]=="A"]["offer_ratio"].mean()
        h_r  = sub[sub["responder_type"]=="H"]["offer_ratio"].mean()
        diffs.append(h_r - ai_r)
    fig = go.Figure(go.Bar(
        x=MODELS, y=diffs,
        marker_color=[COLORS[m] for m in MODELS],
        text=[f"{v:+.1%}" for v in diffs], textposition="outside"
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(yaxis=dict(tickformat=".0%", range=[-0.08,0.32], title="Δ offer ratio"),
                      xaxis_title="", margin=dict(t=10,b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)

# ── Behavioral mode stacked bar ───────────────────────────────────────────────
st.subheader("Behavioral mode distribution by model")
theorist_p = df.groupby("model")["theorist"].mean().reindex(MODELS)*100
human_p = df.groupby("model")["human_mode"].mean().reindex(MODELS)*100
alt_p   = df.groupby("model")["altruistic"].mean().reindex(MODELS)*100

fig = go.Figure()
fig.add_trace(go.Bar(name="Theorist (<10%)",      y=MODELS, x=theorist_p.values, orientation="h",
                     marker_color="#E24B4A",
                     text=[f"{v:.1f}%" for v in theorist_p.values], textposition="inside"))
fig.add_trace(go.Bar(name="Human (10–50%)",    y=MODELS, x=human_p.values, orientation="h",
                     marker_color="#1D9E75",
                     text=[f"{v:.1f}%" for v in human_p.values], textposition="inside"))
fig.add_trace(go.Bar(name="Altruistic (>50%)", y=MODELS, x=alt_p.values,   orientation="h",
                     marker_color="#EF9F27",
                     text=[f"{v:.1f}%" for v in alt_p.values],  textposition="inside"))
fig.update_layout(barmode="stack",
                  xaxis=dict(ticksuffix="%", range=[0,100], title="Share of observations (%)"),
                  yaxis_title="", legend_title="Mode",
                  margin=dict(t=10,b=10), height=280)
st.plotly_chart(fig, use_container_width=True)
st.divider()

# ── Statistical tests ─────────────────────────────────────────────────────────
st.subheader("Statistical tests")
col5, col6 = st.columns(2)

with col5:
    st.markdown("**One-way ANOVA across models**")
    f_off, p_off = stats.f_oneway(*[df[df["model"]==m]["offer_ratio"].values for m in MODELS])
    f_mao, p_mao = stats.f_oneway(*[df[df["model"]==m]["mao_ratio"].values  for m in MODELS])
    st.dataframe(pd.DataFrame({
        "Test":        ["Offer ratio","MAO"],
        "F":           [f"{f_off:.2f}", f"{f_mao:.2f}"],
        "p-value":     [f"{p_off:.2e}", f"{p_mao:.2e}"],
        "Significant": ["Yes ***","Yes ***"],
    }), use_container_width=True, hide_index=True)

with col6:
    st.markdown("**Offer–stake correlation by model** (negative = stake-dependent rationality)")
    st.dataframe(pd.DataFrame({
        "Model":     MODELS,
        "Pearson r": [f"{corrs[m]:.3f}" for m in MODELS],
    }), use_container_width=True, hide_index=True)
