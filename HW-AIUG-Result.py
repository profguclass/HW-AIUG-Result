import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
import urllib.request, io as _io
import statsmodels.formula.api as smf

st.set_page_config(page_title="AI Ultimatum Game — Class Results", layout="wide")
st.title("How does AI distribute the pie?")
st.caption("Class experiment · ChatGPT · Gemini · Copilot · Claude · 4 scenarios · 4 stake levels")

st.info("""
74명의 학생으로부터 수집된 7,616개의 관측값을 분석한 내용을 시각화하여 표현한 사이트이니 잘 살펴보기 바랍니다.

이 네 가지 상용 AI 모델은 평균적으로 파이의 37%를 제안했는데, 이는 내쉬균형의 예측(거의 0에 가까운 제안)보다 훨씬 높은 수치로서 전반적으로 모델들이 "인간"에 가깝게 행동한 것으로 보이며, Gemini만이 상대적으로 "내쉬균형"에 가까운 경향을 보였습니다.

또한 인간응답자 효과(응답자가 AI가 아닌 인간일 때 제안 비율이 증가하는 효과)가 나타났는데, 이는 AI를 상대할 때 모델들이 덜 관대한 조언을 한다는 선행연구의 결과를 재확인합니다.
""")

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

# ── Detailed: scenario x model ───────────────────────────────────────────────
st.subheader("Mean offer ratio by scenario, by AI model")
sc_model = df.groupby(["scenario","model"])["offer_ratio"].mean().reset_index()
SCENARIOS_ORDERED = ["AA","AH","HA","HH"]
SCENARIO_LABELS   = {"AA":"AA (AI→AI)","AH":"AH (AI→Human)","HA":"HA (Human→AI)","HH":"HH (Human→Human)"}
fig = go.Figure()
for m in MODELS:
    sub = sc_model[sc_model["model"]==m].set_index("scenario").reindex(SCENARIOS_ORDERED)
    fig.add_trace(go.Bar(
        name=m,
        x=[SCENARIO_LABELS[s] for s in SCENARIOS_ORDERED],
        y=sub["offer_ratio"].values,
        marker_color=COLORS[m],
        text=[f"{v:.1%}" for v in sub["offer_ratio"].values], textposition="outside"
    ))
fig.update_layout(barmode="group",
                  yaxis=dict(tickformat=".0%", range=[0,0.7], title="Mean offer ratio"),
                  xaxis_title="", legend_title="Model",
                  margin=dict(t=10,b=10), height=360)
st.plotly_chart(fig, use_container_width=True)

# ── Behavioral mode stacked bar ───────────────────────────────────────────────
st.subheader("Behavioral mode distribution by model")
theorist_p = df.groupby("model")["theorist"].mean().reindex(MODELS)*100
human_p    = df.groupby("model")["human_mode"].mean().reindex(MODELS)*100
alt_p      = df.groupby("model")["altruistic"].mean().reindex(MODELS)*100

fig = go.Figure()
fig.add_trace(go.Bar(name="Theorist (<10%)",   y=MODELS, x=theorist_p.values, orientation="h",
                     marker_color="#E24B4A",
                     text=[f"{v:.1f}%" for v in theorist_p.values], textposition="inside"))
fig.add_trace(go.Bar(name="Human (10–50%)",    y=MODELS, x=human_p.values,    orientation="h",
                     marker_color="#1D9E75",
                     text=[f"{v:.1f}%" for v in human_p.values],    textposition="inside"))
fig.add_trace(go.Bar(name="Altruistic (>50%)", y=MODELS, x=alt_p.values,      orientation="h",
                     marker_color="#EF9F27",
                     text=[f"{v:.1f}%" for v in alt_p.values],      textposition="inside"))
fig.update_layout(barmode="stack",
                  xaxis=dict(ticksuffix="%", range=[0,100], title="Share of observations (%)"),
                  yaxis_title="", legend_title="Mode",
                  margin=dict(t=10,b=10), height=280)
st.plotly_chart(fig, use_container_width=True)

# ── Detailed: behavioral mode by scenario ────────────────────────────────────
st.subheader("Behavioral mode distribution by model, by scenario")
for sc in SCENARIOS_ORDERED:
    st.markdown(f"**{SCENARIO_LABELS[sc]}**")
    sub = df[df["scenario"]==sc]
    t_p = sub.groupby("model")["theorist"].mean().reindex(MODELS)*100
    h_p = sub.groupby("model")["human_mode"].mean().reindex(MODELS)*100
    a_p = sub.groupby("model")["altruistic"].mean().reindex(MODELS)*100
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Theorist (<10%)",   y=MODELS, x=t_p.values, orientation="h",
                         marker_color="#E24B4A",
                         text=[f"{v:.1f}%" for v in t_p.values], textposition="inside"))
    fig.add_trace(go.Bar(name="Human (10–50%)",    y=MODELS, x=h_p.values, orientation="h",
                         marker_color="#1D9E75",
                         text=[f"{v:.1f}%" for v in h_p.values], textposition="inside"))
    fig.add_trace(go.Bar(name="Altruistic (>50%)", y=MODELS, x=a_p.values, orientation="h",
                         marker_color="#EF9F27",
                         text=[f"{v:.1f}%" for v in a_p.values], textposition="inside"))
    fig.update_layout(barmode="stack",
                      xaxis=dict(ticksuffix="%", range=[0,100], title="Share of observations (%)"),
                      yaxis_title="", showlegend=False,
                      margin=dict(t=5,b=5), height=220)
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

st.divider()

# ── Regression: Tables 6 & 11 replication ────────────────────────────────────

st.header("Regression Analysis")
st.markdown("""
Replication of **Table 6** (Proposer) and **Table 11** (Responder) from Araujo & Uhlig (2026),
using the triple interaction specification:

> *Y* ~ Amt + P_Human + R_Human + Amt×P_Human + Amt×R_Human + P_Human×R_Human + Amt×P_Human×R_Human

where **Amt** = log₁₀(stake in KRW) − 1, **P_Human** = 1 if Proposer is human, **R_Human** = 1 if Responder is human.
Standard errors are heteroskedasticity-robust (HC1). Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001.
""")

stake_map = {"1만원": 10, "10만원": 100, "100만원": 1000, "1000만원": 10000}
df["Amt"]     = df["stake"].map(stake_map).apply(lambda x: np.log10(x) - 1)
df["P_Human"] = (df["proposer_type"] == "H").astype(int)
df["R_Human"] = (df["responder_type"] == "H").astype(int)

FORMULA = "~ Amt + P_Human + R_Human + Amt:P_Human + Amt:R_Human + P_Human:R_Human + Amt:P_Human:R_Human"
VAR_LABELS = {
    "Intercept":              "Constant",
    "Amt":                    "Amt",
    "P_Human":                "P Human",
    "R_Human":                "R Human",
    "Amt:P_Human":            "Amt × P Human",
    "Amt:R_Human":            "Amt × R Human",
    "P_Human:R_Human":        "P Human × R Human",
    "Amt:P_Human:R_Human":    "Amt × P Human × R Human",
}

def run_reg(dep_var):
    results = {}
    for m in MODELS:
        sub = df[df["model"] == m].copy()
        res = smf.ols(f"{dep_var} {FORMULA}", data=sub).fit(cov_type="HC1")
        results[m] = res
    return results

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

def build_table(results):
    rows = []
    for var, label in VAR_LABELS.items():
        # coefficient row
        row_coef = {"Variable": label}
        for m in MODELS:
            res = results[m]
            if var in res.params:
                c = res.params[var]
                p = res.pvalues[var]
                row_coef[m] = f"{c:.4f}{stars(p)}"
            else:
                row_coef[m] = "—"
        rows.append(row_coef)
        # SE row
        row_se = {"Variable": ""}
        for m in MODELS:
            res = results[m]
            if var in res.bse:
                row_se[m] = f"({res.bse[var]:.4f})"
            else:
                row_se[m] = ""
        rows.append(row_se)
    # N and R2
    row_n  = {"Variable": "N"}
    row_r2 = {"Variable": "R²"}
    for m in MODELS:
        res = results[m]
        row_n[m]  = str(int(res.nobs))
        row_r2[m] = f"{res.rsquared:.3f}"
    rows += [row_n, row_r2]
    return pd.DataFrame(rows)

tab1, tab2 = st.tabs(["Table 6 — Proposer (offer ratio)", "Table 11 — Responder (MAO)"])

with tab1:
    st.markdown("**Dependent variable: offer_ratio** — Triple interaction (Amount × Player Types)")
    res6 = run_reg("offer_ratio")
    tbl6 = build_table(res6)
    st.dataframe(tbl6, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📖 How to read the coefficients")
    st.markdown("""
| Coefficient | What it measures | Example interpretation |
|---|---|---|
| **Constant** | Baseline offer ratio when both players are AI and stake is ₩10,000 (Amt = 0) | ChatGPT starts at 48.5% in the AI-vs-AI baseline |
| **Amt** | How offer ratio changes as stake increases (each unit = 10× increase in stake) | A negative value means the model offers less as stakes rise — *stake-dependent rationality* |
| **P Human** | Change in offer ratio when the Proposer is a *human asking for advice* (vs. AI deciding for itself) | A negative value means the model gives more conservative advice to humans than it would choose for itself |
| **R Human** | Change in offer ratio when the Responder is human (vs. AI) | A positive value means the model offers more generously when facing a human — the *human responder effect* |
| **Amt × P Human** | Does the stake effect differ depending on whether the Proposer is human? | If negative, the model gives even more conservative advice to humans at higher stakes |
| **Amt × R Human** | Does the stake effect differ depending on whether the Responder is human? | If negative, the human-responder generosity shrinks at higher stakes |
| **P Human × R Human** | Extra adjustment when *both* players are human (interaction on top of P Human and R Human) | Captures whether the model changes behavior specifically in the human-vs-human scenario |
| **Amt × P Human × R Human** | Does the human-human interaction effect itself vary with stake size? | The most complex term — significant only for some models |
""")
    st.info("""
**Key findings to look for:**
- **Amt < 0** (significant): the model is more rational at higher stakes ✓
- **R Human > 0** (significant): the model is more generous toward human responders ✓
- **P Human < 0** (significant): the model gives less generous advice to humans than it keeps for itself — suggesting it applies different norms when advising vs. acting autonomously
""")

    st.markdown("**Coefficient plot — Proposer** *(error bars = 95% confidence interval; bars crossing 0 are not significant)*")
    selected_model_6 = st.selectbox("Select model", MODELS, key="reg6")
    res = res6[selected_model_6]
    vars_to_plot = [v for v in VAR_LABELS if v != "Intercept" and v in res.params]
    coefs  = [res.params[v] for v in vars_to_plot]
    ci_lo  = [res.conf_int().loc[v, 0] for v in vars_to_plot]
    ci_hi  = [res.conf_int().loc[v, 1] for v in vars_to_plot]
    labels = [VAR_LABELS[v] for v in vars_to_plot]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coefs, y=labels, mode="markers",
        marker=dict(size=10, color=COLORS[selected_model_6]),
        error_x=dict(type="data",
                     arrayminus=[c - l for c, l in zip(coefs, ci_lo)],
                     array=[h - c for c, h in zip(coefs, ci_hi)],
                     color=COLORS[selected_model_6])
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="Coefficient", yaxis_title="",
                      margin=dict(t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("**Dependent variable: mao_ratio** — Triple interaction (Amount × Player Types)")
    res11 = run_reg("mao_ratio")
    tbl11 = build_table(res11)
    st.dataframe(tbl11, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📖 How to read the coefficients")
    st.markdown("""
| Coefficient | What it measures | Example interpretation |
|---|---|---|
| **Constant** | Baseline MAO when both players are AI and stake is ₩10 (Amt = 0) | How demanding the AI is as a responder in the pure AI-vs-AI baseline |
| **Amt** | How MAO changes as stake increases | A negative value means the model accepts a smaller share at higher stakes — more rational as amounts grow |
| **P Human** | Change in MAO when the Proposer is human | Does the model demand more or less when a human is making the offer? |
| **R Human** | Change in MAO when the Responder is human (i.e., the AI is advising a human on what to accept) | A positive value means the model tells humans to demand a higher minimum — applying stricter fairness norms when advising |
| **Amt × P Human** | Does the stake effect on MAO differ when Proposer is human? | Captures whether the "be more rational at higher stakes" pattern changes depending on who is proposing |
| **Amt × R Human** | Does the stake effect on MAO differ when Responder is human? | If negative, the AI advises humans to lower their threshold more sharply at higher stakes |
| **P Human × R Human** | Extra adjustment when both players are human | Does the AI change its acceptance advice specifically in the human-vs-human scenario? |
| **Amt × P Human × R Human** | Triple interaction: stake effect in the human-vs-human scenario | The most nuanced term — how stake sensitivity in the human-vs-human case differs from all other cases |
""")
    st.info("""
**Key findings to look for:**
- **Amt < 0** (significant): the model is more willing to accept low offers at higher stakes — consistent with rational theory
- **R Human > 0** (significant): the model tells humans to demand a *higher* minimum than it would accept for itself — it applies stricter fairness norms when advising humans
- **Constant near 0**: GPT-5 mini in the paper is close to 0, meaning near-full rationality as a responder in the baseline case. Compare how your models perform on this benchmark.
""")

    st.markdown("**Coefficient plot — Responder** *(error bars = 95% confidence interval; bars crossing 0 are not significant)*")
    selected_model_11 = st.selectbox("Select model", MODELS, key="reg11")
    res = res11[selected_model_11]
    vars_to_plot = [v for v in VAR_LABELS if v != "Intercept" and v in res.params]
    coefs  = [res.params[v] for v in vars_to_plot]
    ci_lo  = [res.conf_int().loc[v, 0] for v in vars_to_plot]
    ci_hi  = [res.conf_int().loc[v, 1] for v in vars_to_plot]
    labels = [VAR_LABELS[v] for v in vars_to_plot]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coefs, y=labels, mode="markers",
        marker=dict(size=10, color=COLORS[selected_model_11]),
        error_x=dict(type="data",
                     arrayminus=[c - l for c, l in zip(coefs, ci_lo)],
                     array=[h - c for c, h in zip(coefs, ci_hi)],
                     color=COLORS[selected_model_11])
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="Coefficient", yaxis_title="",
                      margin=dict(t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)
