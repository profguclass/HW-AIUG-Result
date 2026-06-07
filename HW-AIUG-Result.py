import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
import urllib.request, io as _io
import statsmodels.formula.api as smf

st.set_page_config(page_title="AI Ultimatum Game — Experiment Result", layout="wide")
st.title("How does AI distribute the pie?")
st.caption("경기대학교 경제학전공 '게임이론' 실험과제의 종합분석 (2026년 6월) · 4 AIs(ChatGPT,Gemini,Copilot,Claude) · 4 scenarios · 4 stake levels")
st.info("""총 74명의 학생들이 수행한 7,616개의 실험관측값을 종합하여 분석한 결과, 네 가지 범용 AI모델은 평균적으로 37%를 제안(Offer)했고 최소수용제안(MAO)은 12%였다. 이는 내쉬균형의 예측(거의 0에 가까운 제안)보다 훨씬 높은 수치로서 전반적으로 AI모델들이 "인간"에 가깝게 행동한 것을 의미한다(Gemini만이 상대적으로 "내쉬균형"에 가까운 경향을 보였다).
또한 인간응답자 효과(응답자가 AI가 아닌 인간일 때 제안비율이 증가하는 효과)가 나타났는데, 이는 AI모델들이 AI를 상대할 때 덜 관대한 조언을 한다는 선행연구의 결과를 재확인하고 있다. 그리고 선행연구에서처럼 금액크기(stake)에 따른 조정이 확인되었다. 네 모델 모두 금액이 ₩10,000에서 ₩10,000,000으로 증가할수록 제안비율을 낮추는 경향을 보였다. 
끝으로 네 개의 AI모델의 평균 제안비율 간 차이는 통계적으로 무시할 수 없는 수준인데, 이는 어떤 AI모델을 사용하느냐가 파이를 얼마나 나눠주느냐에 실질적이고 유의미한 영향을 미친다는 것을 의미한다.""")

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
    st.plotly_chart(fig, use_container_width=True, key="chart_mean_offer")

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
    st.plotly_chart(fig, use_container_width=True, key="chart_mean_mao")

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
st.plotly_chart(fig, use_container_width=True, key="chart_sd_variability")

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
st.plotly_chart(fig, use_container_width=True, key="chart_offer_by_stake")

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
    st.plotly_chart(fig, use_container_width=True, key="chart_offer_by_scenario")

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
    st.plotly_chart(fig, use_container_width=True, key="chart_human_responder_effect")

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
st.plotly_chart(fig, use_container_width=True, key="chart_scenario_by_model")

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
st.plotly_chart(fig, use_container_width=True, key="chart_behavioral_mode")

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
    st.plotly_chart(fig, use_container_width=True, key=f"chart_behavioral_by_scenario_{sc}")

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

# ── Regression Analysis ───────────────────────────────────────────────────────

st.header("Regression Analysis")
st.markdown("""
Inspired by Araujo & Uhlig (2026), we run OLS regressions for each AI model.
All specifications include:
- **Amt** = log₁₀(stake in KRW) − 1
- **P_Human** = 1 if Proposer is human, 0 if AI
- **R_Human** = 1 if Responder is human, 0 if AI

Standard errors are heteroskedasticity-robust (HC1). Significance: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001.
""")

stake_map = {"1만원": 1, "10만원": 10, "100만원": 100, "1000만원": 1000}
df["Amt"]     = df["stake"].map(stake_map).apply(lambda x: np.log10(x))
df["P_Human"] = (df["proposer_type"] == "H").astype(int)
df["R_Human"] = (df["responder_type"] == "H").astype(int)

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

def run_reg(dep_var, formula):
    results = {}
    for m in MODELS:
        sub = df[df["model"] == m].copy()
        results[m] = smf.ols(f"{dep_var} {formula}", data=sub).fit(cov_type="HC1")
    return results

def build_table(results, var_labels):
    rows = []
    for var, label in var_labels.items():
        row_coef = {"Variable": label}
        row_se   = {"Variable": ""}
        for m in MODELS:
            res = results[m]
            if var in res.params:
                row_coef[m] = f"{res.params[var]:.4f}{stars(res.pvalues[var])}"
                row_se[m]   = f"({res.bse[var]:.4f})"
            else:
                row_coef[m] = "—"
                row_se[m]   = ""
        rows += [row_coef, row_se]
    row_n  = {"Variable": "N"}
    row_r2 = {"Variable": "R²"}
    for m in MODELS:
        row_n[m]  = str(int(results[m].nobs))
        row_r2[m] = f"{results[m].rsquared:.3f}"
    rows += [row_n, row_r2]
    return pd.DataFrame(rows)

# ── Simple (main effects) tables ──────────────────────────────────────────────
st.subheader("Simple model: main effects only")
st.markdown("""
> *Y* ~ **Constant** + **Amt** + **P_Human** + **R_Human**

This is the easiest specification to interpret — no interaction terms.
""")

FORMULA_SIMPLE = "~ Amt + P_Human + R_Human"
VAR_LABELS_SIMPLE = {
    "Intercept": "Constant",
    "Amt":       "Amt",
    "P_Human":   "P Human",
    "R_Human":   "R Human",
}

COEF_EXPLAIN = """
#### 📖 How to read the coefficients

| Coefficient | What it measures | Positive value means… | Negative value means… |
|---|---|---|---|
| **Constant** | Baseline Y when both players are AI and stake = ₩1만원 (Amt = 0) | High baseline generosity / acceptance threshold | Low baseline — close to rational benchmark |
| **Amt** | Change in Y for each 10× increase in stake | More generous / demanding at higher stakes | More rational (less generous / more accepting) at higher stakes |
| **P Human** | Change in Y when the AI is *advising a human* vs. acting for itself | AI gives more generous advice to humans | AI gives more conservative advice to humans than it keeps for itself |
| **R Human** | Change in Y when the Responder is human vs. AI | AI is more generous toward / demands more from humans | AI treats human and AI opponents similarly or favors AI |
"""

stab1, stab2 = st.tabs(["Proposer (offer ratio)", "Responder (MAO)"])

with stab1:
    st.markdown("**Dependent variable: offer_ratio**")
    res_s6 = run_reg("offer_ratio", FORMULA_SIMPLE)
    st.dataframe(build_table(res_s6, VAR_LABELS_SIMPLE), use_container_width=True, hide_index=True)
    st.markdown(COEF_EXPLAIN)
    st.info("""
**Key findings to look for:**
- **Amt < 0** ✓ Models offer less as stakes rise (*stake-dependent rationality*)
- **R Human > 0** ✓ Models are more generous when the responder is human (*human responder effect*)
- **P Human < 0** Models give more conservative advice to humans than they choose for themselves
""")

with stab2:
    st.markdown("**Dependent variable: mao_ratio**")
    res_s11 = run_reg("mao_ratio", FORMULA_SIMPLE)
    st.dataframe(build_table(res_s11, VAR_LABELS_SIMPLE), use_container_width=True, hide_index=True)
    st.markdown(COEF_EXPLAIN.replace(
        "More generous / demanding at higher stakes", "Demands higher minimum at higher stakes"
    ).replace(
        "More rational (less generous / more accepting) at higher stakes",
        "More rational — accepts smaller shares at higher stakes"
    ).replace(
        "AI gives more generous advice to humans", "AI advises humans to demand more"
    ).replace(
        "AI gives more conservative advice to humans than it keeps for itself",
        "AI advises humans to accept lower offers than it would itself"
    ).replace(
        "AI is more generous toward / demands more from humans",
        "AI sets a higher acceptance threshold when advising humans"
    ).replace(
        "AI treats human and AI opponents similarly or favors AI",
        "AI sets a lower or equal threshold regardless of who proposes"
    ))
    st.info("""
**Key findings to look for:**
- **Amt < 0** ✓ Models accept smaller shares at higher stakes (*stake-dependent rationality*)
- **R Human > 0** ✓ Models tell humans to demand more — stricter fairness norms when advising
- **Constant near 0**: the model behaves close to the rational benchmark as a responder
""")

st.divider()

# ── Full interaction tables ────────────────────────────────────────────────────
st.subheader("Full model: triple interaction (Amount × Player Types)")
st.markdown("""
> *Y* ~ Amt + P_Human + R_Human + Amt×P_Human + Amt×R_Human + P_Human×R_Human + Amt×P_Human×R_Human

This replicates **Table 6** (Proposer) and **Table 11** (Responder) from Araujo & Uhlig (2026).
The interaction terms capture whether the effects of stake size and player type *depend on each other*.
""")

FORMULA_FULL = "~ Amt + P_Human + R_Human + Amt:P_Human + Amt:R_Human + P_Human:R_Human + Amt:P_Human:R_Human"
VAR_LABELS_FULL = {
    "Intercept":           "Constant",
    "Amt":                 "Amt",
    "P_Human":             "P Human",
    "R_Human":             "R Human",
    "Amt:P_Human":         "Amt × P Human",
    "Amt:R_Human":         "Amt × R Human",
    "P_Human:R_Human":     "P Human × R Human",
    "Amt:P_Human:R_Human": "Amt × P Human × R Human",
}

tab1, tab2 = st.tabs(["Table 6 — Proposer (offer ratio)", "Table 11 — Responder (MAO)"])

# ── shared helper functions ───────────────────────────────────────────────────
from scipy import stats as scipy_stats

def me_P(res, amt, r):
    c = np.array([0, 0, 1, 0, amt, 0, r, amt*r])
    me = c @ res.params.values
    se = np.sqrt(c @ res.cov_params().values @ c)
    pv = 2 * scipy_stats.t.sf(abs(me/se), df=res.df_resid)
    return me, se, pv

def me_R(res, amt, p):
    c = np.array([0, 0, 0, 1, 0, amt, p, amt*p])
    me = c @ res.params.values
    se = np.sqrt(c @ res.cov_params().values @ c)
    pv = 2 * scipy_stats.t.sf(abs(me/se), df=res.df_resid)
    return me, se, pv

AMT_VALS   = [0, 1, 2, 3]
AMT_LABELS = ["₩1만원", "₩10만원", "₩100만원", "₩1000만원"]

def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

def show_me_tables(res_dict, selected, dep="offer"):
    # If "All models", average the marginal effects across all models
    # res_dict is the full {model: result} dict; selected is the chosen model or "All models"

    def get_me_P(amt, r):
        if selected == "All models":
            mes = [me_P(res_dict[m], amt, r)[0] for m in MODELS]
            ses = [me_P(res_dict[m], amt, r)[1] for m in MODELS]
            me_avg = np.mean(mes)
            se_avg = np.sqrt(np.mean([s**2 for s in ses]))  # pooled SE
            pv = 2 * scipy_stats.t.sf(abs(me_avg / se_avg), df=min(res_dict[m].df_resid for m in MODELS))
            return me_avg, se_avg, pv
        else:
            return me_P(res_dict[selected], amt, r)

    def get_me_R(amt, p):
        if selected == "All models":
            mes = [me_R(res_dict[m], amt, p)[0] for m in MODELS]
            ses = [me_R(res_dict[m], amt, p)[1] for m in MODELS]
            me_avg = np.mean(mes)
            se_avg = np.sqrt(np.mean([s**2 for s in ses]))
            pv = 2 * scipy_stats.t.sf(abs(me_avg / se_avg), df=min(res_dict[m].df_resid for m in MODELS))
            return me_avg, se_avg, pv
        else:
            return me_R(res_dict[selected], amt, p)

    # ── ME of P_Human ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### \U0001f522 Total marginal effect of P_Human")
    st.markdown("""
The coefficient on **P_Human** alone does not capture its full effect — it must be combined with its interaction terms:

> **ME(P_Human)** = β(P_Human) + β(Amt×P_Human)·Amt + β(P_Human×R_Human)·R_Human + β(Amt×P_Human×R_Human)·Amt·R_Human

The table below evaluates this total effect at each combination of stake level and responder type, with **delta-method** standard errors.
""")
    rows = []
    for r_val, r_label in [(0, "AI Responder"), (1, "Human Responder")]:
        for amt, slabel in zip(AMT_VALS, AMT_LABELS):
            me, se, pv = get_me_P(amt, r_val)
            s = stars(pv)
            if dep == "offer":
                interp = "More conservative advice to humans" if me < 0 else "More generous advice to humans"
            else:
                interp = "AI advises humans to accept lower offers" if me < 0 else "AI advises humans to demand more"
            rows.append({
                "Responder type": r_label,
                "Stake":          slabel,
                "ME of P_Human":  f"{me:+.4f}{s}",
                "Std. Error":     f"({se:.4f})",
                "p-value":        f"{pv:.3f}",
                "Interpretation": interp
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── ME of R_Human ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### \U0001f522 Total marginal effect of R_Human")
    st.markdown("""
> **ME(R_Human)** = β(R_Human) + β(Amt×R_Human)·Amt + β(P_Human×R_Human)·P_Human + β(Amt×P_Human×R_Human)·Amt·P_Human
""")
    rows2 = []
    for p_val, p_label in [(0, "AI Proposer"), (1, "Human Proposer")]:
        for amt, slabel in zip(AMT_VALS, AMT_LABELS):
            me, se, pv = get_me_R(amt, p_val)
            s = stars(pv)
            if dep == "offer":
                interp = "More generous toward human responder" if me > 0 else "Less generous toward human responder"
            else:
                interp = "Higher acceptance threshold when advising humans" if me > 0 else "Lower threshold when advising humans"
            rows2.append({
                "Proposer type":  p_label,
                "Stake":          slabel,
                "ME of R_Human":  f"{me:+.4f}{s}",
                "Std. Error":     f"({se:.4f})",
                "p-value":        f"{pv:.3f}",
                "Interpretation": interp
            })
    st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)
    st.caption("Significance: * p<0.05  ** p<0.01  *** p<0.001. Standard errors via delta method (HC1).")

# ── TAB 1: Proposer ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("**Dependent variable: offer_ratio**")
    res6 = run_reg("offer_ratio", FORMULA_FULL)
    st.dataframe(build_table(res6, VAR_LABELS_FULL), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### \U0001f4d6 How to read the coefficients")
    st.markdown("""
| Coefficient | What it measures | Example interpretation |
|---|---|---|
| **Constant** | Baseline offer ratio when both players are AI and stake = ₩1만원 (Amt = 0) | The AI\'s generosity in the pure AI-vs-AI baseline |
| **Amt** | Change per 10× increase in stake | Negative = more rational at higher stakes (*stake-dependent rationality*) |
| **P Human** | Baseline change when Proposer is human (at Amt=0, R=AI) | Negative = AI gives more conservative advice to humans than it chooses for itself |
| **R Human** | Baseline change when Responder is human (at Amt=0, P=AI) | Positive = AI offers more generously toward humans (*human responder effect*) |
| **Amt \u00d7 P Human** | Does the stake effect change when Proposer is human? | If negative, AI gives even more conservative advice at higher stakes |
| **Amt \u00d7 R Human** | Does the stake effect change when Responder is human? | If negative, the human-responder generosity shrinks at higher stakes |
| **P Human \u00d7 R Human** | Extra shift when *both* players are human (at Amt=0) | Specific adjustment for the human-vs-human scenario |
| **Amt \u00d7 P Human \u00d7 R Human** | Does the human-human adjustment change with stake? | The most complex term — captures stake sensitivity specific to human-vs-human |
""")
    st.info("""
**Key findings to look for:**
- **Amt < 0** (significant): AI is more rational at higher stakes \u2713
- **R Human > 0** (significant): AI is more generous toward human responders \u2713
- **P Human < 0** (significant): AI gives more conservative advice to humans than it acts for itself
\n\u26a0\ufe0f The coefficient on P_Human alone is only the effect *at ₩1만원 with an AI responder*. See the marginal effects tables below for the full picture.
""")

    st.markdown("**Coefficient plot** *(error bars = 95% CI; bars crossing 0 are not significant)*")
    sel6_plot = st.selectbox("Select model", MODELS, key="coefplot_proposer")
    res = res6[sel6_plot]
    vars_to_plot = [v for v in VAR_LABELS_FULL if v != "Intercept" and v in res.params]
    coefs = [res.params[v] for v in vars_to_plot]
    ci_lo = [res.conf_int().loc[v, 0] for v in vars_to_plot]
    ci_hi = [res.conf_int().loc[v, 1] for v in vars_to_plot]
    labels = [VAR_LABELS_FULL[v] for v in vars_to_plot]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coefs, y=labels, mode="markers",
        marker=dict(size=10, color=COLORS[sel6_plot]),
        error_x=dict(type="data",
                     arrayminus=[c - l for c, l in zip(coefs, ci_lo)],
                     array=[h - c for c, h in zip(coefs, ci_hi)],
                     color=COLORS[sel6_plot])
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="Coefficient", yaxis_title="",
                      margin=dict(t=10, b=10), height=340)
    st.plotly_chart(fig, use_container_width=True, key="chart_coef_proposer")

    st.markdown("---")
    st.markdown("#### \U0001f4ca Marginal effects (accounting for all interaction terms)")
    st.markdown("Select a model to see the **total** effect of each player-type variable across all stake levels and opponent types.")
    sel6_me = st.selectbox("Select model for marginal effects", ["All models"] + MODELS, key="me6")
    show_me_tables(res6, sel6_me, dep="offer")

# ── TAB 2: Responder ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("**Dependent variable: mao_ratio**")
    res11 = run_reg("mao_ratio", FORMULA_FULL)
    st.dataframe(build_table(res11, VAR_LABELS_FULL), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### \U0001f4d6 How to read the coefficients")
    st.markdown("""
| Coefficient | What it measures | Example interpretation |
|---|---|---|
| **Constant** | Baseline MAO when both players are AI and stake = ₩1만원 (Amt = 0) | How demanding the AI is in the pure AI-vs-AI baseline |
| **Amt** | Change per 10\u00d7 increase in stake | Negative = accepts smaller shares at higher stakes — more rational |
| **P Human** | Baseline change in MAO when Proposer is human (at Amt=0, R=AI) | Does the model demand more or less from a human proposer? |
| **R Human** | Baseline change in MAO when Responder is human (at Amt=0, P=AI) | Positive = AI tells humans to demand a higher minimum |
| **Amt \u00d7 P Human** | Does the stake effect change when Proposer is human? | Whether "be more rational at higher stakes" shifts with proposer type |
| **Amt \u00d7 R Human** | Does the stake effect change when Responder is human? | Negative = AI advises humans to lower their threshold more at higher stakes |
| **P Human \u00d7 R Human** | Extra shift when both players are human (at Amt=0) | AI-specific adjustment for the human-vs-human scenario |
| **Amt \u00d7 P Human \u00d7 R Human** | Does the human-human adjustment change with stake? | Stake sensitivity specific to human-vs-human |
""")
    st.info("""
**Key findings to look for:**
- **Amt < 0** (significant): AI accepts lower shares at higher stakes \u2713
- **R Human > 0** (significant): AI tells humans to demand more — stricter fairness norms when advising \u2713
- **Constant near 0**: close to the rational benchmark as a responder in the AI-vs-AI baseline
\n\u26a0\ufe0f The coefficient on P_Human alone is only the effect *at ₩1만원 with an AI responder*. See the marginal effects tables below for the full picture.
""")

    st.markdown("**Coefficient plot** *(error bars = 95% CI; bars crossing 0 are not significant)*")
    sel11_plot = st.selectbox("Select model", MODELS, key="coefplot_responder")
    res = res11[sel11_plot]
    vars_to_plot = [v for v in VAR_LABELS_FULL if v != "Intercept" and v in res.params]
    coefs = [res.params[v] for v in vars_to_plot]
    ci_lo = [res.conf_int().loc[v, 0] for v in vars_to_plot]
    ci_hi = [res.conf_int().loc[v, 1] for v in vars_to_plot]
    labels = [VAR_LABELS_FULL[v] for v in vars_to_plot]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coefs, y=labels, mode="markers",
        marker=dict(size=10, color=COLORS[sel11_plot]),
        error_x=dict(type="data",
                     arrayminus=[c - l for c, l in zip(coefs, ci_lo)],
                     array=[h - c for c, h in zip(coefs, ci_hi)],
                     color=COLORS[sel11_plot])
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="Coefficient", yaxis_title="",
                      margin=dict(t=10, b=10), height=340)
    st.plotly_chart(fig, use_container_width=True, key="chart_coef_responder")

    st.markdown("---")
    st.markdown("#### \U0001f4ca Marginal effects (accounting for all interaction terms)")
    st.markdown("Select a model to see the **total** effect of each player-type variable across all stake levels and opponent types.")
    sel11_me = st.selectbox("Select model for marginal effects", ["All models"] + MODELS, key="me11")
    show_me_tables(res11, sel11_me, dep="mao")
