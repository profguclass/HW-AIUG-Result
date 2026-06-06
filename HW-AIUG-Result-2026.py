import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import io

st.set_page_config(page_title="AI Ultimatum Game — Class Results", layout="wide")

st.title("How does AI distribute the pie?")
st.caption("Class experiment: ChatGPT · Gemini · Copilot · Claude across 4 scenarios and 4 stake levels")

uploaded = st.file_uploader("Upload the class Excel file (HW-AIUG-ed.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Please upload the Excel file to see the analysis.")
    st.stop()

df = pd.read_excel(uploaded)

stake_order = {"1만원": 1, "10만원": 2, "100만원": 3, "1000만원": 4}
df["stake_rank"] = df["stake"].map(stake_order)
df["altruistic"] = df["offer_ratio"] > 0.5
df["spock"] = df["offer_ratio"] < 0.1
df["human_mode"] = (df["offer_ratio"] >= 0.1) & (df["offer_ratio"] <= 0.5)

MODELS = ["ChatGPT", "Gemini", "Copilot", "Claude"]
COLORS = {"ChatGPT": "#378ADD", "Claude": "#1D9E75", "Copilot": "#D4537E", "Gemini": "#EF9F27"}
STAKES_ORDERED = ["1만원", "10만원", "100만원", "1000만원"]

# ── Overview metrics ─────────────────────────────────────────────────────────
st.subheader("Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Observations", f"{len(df):,}")
c2.metric("Students", df["student_id"].nunique())
c3.metric("Mean offer ratio", f"{df['offer_ratio'].mean():.1%}")
c4.metric("Mean MAO", f"{df['mao_ratio'].mean():.1%}")
c5.metric("Altruistic offers (>50%)", f"{df['altruistic'].mean():.1%}")

st.divider()

# ── Model-level summary table ────────────────────────────────────────────────
st.subheader("Model-level summary")

summary = df.groupby("model").agg(
    n=("offer_ratio", "count"),
    mean_offer=("offer_ratio", "mean"),
    sd_offer=("offer_ratio", "std"),
    mean_mao=("mao_ratio", "mean"),
    pct_altruistic=("altruistic", "mean"),
    pct_spock=("spock", "mean"),
    pct_human=("human_mode", "mean"),
).reindex(MODELS)

corrs = {m: df[df["model"] == m]["offer_ratio"].corr(df[df["model"] == m]["stake_rank"]) for m in MODELS}
summary["offer_stake_corr"] = pd.Series(corrs)

fmt = {
    "mean_offer": "{:.1%}", "sd_offer": "{:.1%}", "mean_mao": "{:.1%}",
    "pct_altruistic": "{:.1%}", "pct_spock": "{:.1%}", "pct_human": "{:.1%}",
    "offer_stake_corr": "{:.3f}",
}
display_summary = summary.copy()
for col, f in fmt.items():
    display_summary[col] = display_summary[col].apply(lambda x: f.format(x))

display_summary.columns = ["N", "Mean offer", "SD offer", "Mean MAO",
                            "% Altruistic (>50%)", "% Spock (<10%)", "% Human (10–50%)",
                            "Offer–stake corr."]
st.dataframe(display_summary, use_container_width=True)

st.divider()

# ── Charts ───────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

# Mean offer by model
with col_a:
    st.subheader("Mean offer ratio by model")
    means = df.groupby("model")["offer_ratio"].mean().reindex(MODELS)
    fig = go.Figure(go.Bar(
        x=MODELS, y=means.values,
        marker_color=[COLORS[m] for m in MODELS],
        text=[f"{v:.1%}" for v in means.values], textposition="outside"
    ))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 0.65],
                      yaxis_title="Mean offer ratio", xaxis_title="",
                      margin=dict(t=20, b=20), height=300)
    st.plotly_chart(fig, use_container_width=True)

# Mean MAO by model
with col_b:
    st.subheader("Mean minimum acceptable offer (MAO) by model")
    maos = df.groupby("model")["mao_ratio"].mean().reindex(MODELS)
    fig = go.Figure(go.Bar(
        x=MODELS, y=maos.values,
        marker_color=[COLORS[m] for m in MODELS],
        text=[f"{v:.1%}" for v in maos.values], textposition="outside"
    ))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 0.35],
                      yaxis_title="Mean MAO", xaxis_title="",
                      margin=dict(t=20, b=20), height=300)
    st.plotly_chart(fig, use_container_width=True)

# Offer by model x stake
st.subheader("Mean offer ratio by model and stake level (stake-dependent rationality)")
stake_pivot = df.groupby(["model", "stake"])["offer_ratio"].mean().reset_index()

fig = go.Figure()
for m in MODELS:
    sub = stake_pivot[stake_pivot["model"] == m].set_index("stake").reindex(STAKES_ORDERED)
    fig.add_trace(go.Bar(
        name=m, x=STAKES_ORDERED, y=sub["offer_ratio"].values,
        marker_color=COLORS[m],
        text=[f"{v:.1%}" for v in sub["offer_ratio"].values], textposition="outside"
    ))
fig.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_range=[0, 0.65],
                  yaxis_title="Mean offer ratio", xaxis_title="Stake",
                  legend_title="Model", margin=dict(t=20, b=20), height=340)
st.plotly_chart(fig, use_container_width=True)

col_c, col_d = st.columns(2)

# Offer by scenario
with col_c:
    st.subheader("Mean offer ratio by scenario")
    scenario_means = df.groupby("scenario")["offer_ratio"].mean().reindex(["AA", "AH", "HA", "HH"])
    labels = ["AA (AI→AI)", "AH (AI→Human)", "HA (Human→AI)", "HH (Human→Human)"]
    fig = go.Figure(go.Bar(
        x=labels, y=scenario_means.values,
        marker_color=["#B5D4F4", "#378ADD", "#85B7EB", "#185FA5"],
        text=[f"{v:.1%}" for v in scenario_means.values], textposition="outside"
    ))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 0.6],
                      yaxis_title="Mean offer ratio", xaxis_title="",
                      margin=dict(t=20, b=20), height=320)
    st.plotly_chart(fig, use_container_width=True)

# Human responder effect
with col_d:
    st.subheader("Human responder effect on offer (Δ pp vs AI responder)")
    diffs = []
    for m in MODELS:
        sub = df[df["model"] == m]
        ai_r = sub[sub["responder_type"] == "A"]["offer_ratio"].mean()
        h_r = sub[sub["responder_type"] == "H"]["offer_ratio"].mean()
        diffs.append(h_r - ai_r)
    fig = go.Figure(go.Bar(
        x=MODELS, y=diffs,
        marker_color=["#378ADD", "#EF9F27", "#E24B4A", "#1D9E75"],
        text=[f"{v:+.1%}" for v in diffs], textposition="outside"
    ))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[-0.1, 0.35],
                      yaxis_title="Δ offer ratio", xaxis_title="",
                      shapes=[dict(type="line", x0=-0.5, x1=3.5, y0=0, y1=0,
                                   line=dict(color="gray", width=1, dash="dot"))],
                      margin=dict(t=20, b=20), height=320)
    st.plotly_chart(fig, use_container_width=True)

# Behavioral mode stacked bar
st.subheader("Behavioral mode distribution by model")
spock_pct = df.groupby("model")["spock"].mean().reindex(MODELS) * 100
human_pct = df.groupby("model")["human_mode"].mean().reindex(MODELS) * 100
alt_pct = df.groupby("model")["altruistic"].mean().reindex(MODELS) * 100

fig = go.Figure()
fig.add_trace(go.Bar(name="Spock (<10%)", y=MODELS, x=spock_pct.values,
                     orientation="h", marker_color="#E24B4A",
                     text=[f"{v:.1f}%" for v in spock_pct.values], textposition="inside"))
fig.add_trace(go.Bar(name="Human (10–50%)", y=MODELS, x=human_pct.values,
                     orientation="h", marker_color="#1D9E75",
                     text=[f"{v:.1f}%" for v in human_pct.values], textposition="inside"))
fig.add_trace(go.Bar(name="Altruistic (>50%)", y=MODELS, x=alt_pct.values,
                     orientation="h", marker_color="#EF9F27",
                     text=[f"{v:.1f}%" for v in alt_pct.values], textposition="inside"))
fig.update_layout(barmode="stack", xaxis_ticksuffix="%", xaxis_range=[0, 100],
                  xaxis_title="Share of observations (%)", yaxis_title="",
                  legend_title="Mode", margin=dict(t=20, b=20), height=280)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Inference ────────────────────────────────────────────────────────────────
st.subheader("Statistical tests")

col_e, col_f = st.columns(2)

with col_e:
    st.markdown("**One-way ANOVA across models**")
    groups_offer = [df[df["model"] == m]["offer_ratio"].values for m in MODELS]
    f_off, p_off = stats.f_oneway(*groups_offer)
    groups_mao = [df[df["model"] == m]["mao_ratio"].values for m in MODELS]
    f_mao, p_mao = stats.f_oneway(*groups_mao)
    anova_df = pd.DataFrame({
        "Test": ["Offer ratio across models", "MAO across models"],
        "F statistic": [f"{f_off:.2f}", f"{f_mao:.2f}"],
        "p-value": [f"{p_off:.2e}", f"{p_mao:.2e}"],
        "Significant?": ["Yes ***" if p_off < 0.001 else "No",
                         "Yes ***" if p_mao < 0.001 else "No"],
    })
    st.dataframe(anova_df, use_container_width=True, hide_index=True)

with col_f:
    st.markdown("**Offer–stake correlation by model** (negative = stake-dependent rationality)")
    corr_df = pd.DataFrame({
        "Model": MODELS,
        "Pearson r": [f"{corrs[m]:.3f}" for m in MODELS],
        "Direction": ["↓ More rational at higher stakes" if corrs[m] < 0 else "↑ More generous at higher stakes"
                      for m in MODELS],
    })
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

st.divider()

# ── Raw data explorer ────────────────────────────────────────────────────────
with st.expander("Explore raw data"):
    filter_model = st.multiselect("Filter by model", MODELS, default=MODELS)
    filter_scenario = st.multiselect("Filter by scenario", ["AA", "AH", "HA", "HH"],
                                     default=["AA", "AH", "HA", "HH"])
    filtered = df[(df["model"].isin(filter_model)) & (df["scenario"].isin(filter_scenario))]
    st.dataframe(filtered[["student_id", "model", "scenario", "proposer_type",
                            "responder_type", "stake", "offer_ratio", "mao_ratio"]],
                 use_container_width=True)
    st.caption(f"{len(filtered):,} rows shown")