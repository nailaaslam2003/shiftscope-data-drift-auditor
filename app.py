from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency, ks_2samp

st.set_page_config(page_title="ShiftScope | Data Drift Auditor", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #f6f8fc 0%, #edf2f8 100%); }
[data-testid="stSidebar"] { background: #14213d; }
[data-testid="stSidebar"] * { color: #f7f9ff !important; }
.hero { padding: 1.45rem 1.7rem; border-radius: 20px; color: white;
        background: linear-gradient(110deg, #14213d, #315c79, #39827d); margin-bottom: 1rem; }
.hero h1 { margin: 0; font-size: 2.15rem; }
.hero p { margin: .45rem 0 0; color: #dceaf3; }
.card { background: white; border: 1px solid #e1e8f0; border-radius: 16px;
        padding: 1rem 1.1rem; min-height: 105px; box-shadow: 0 3px 15px #14213d0b; }
.label { color: #63748a; font-size: .84rem; }
.value { color: #172554; font-size: 1.8rem; font-weight: 750; }
.note { color: #718096; font-size: .78rem; }
</style>
""", unsafe_allow_html=True)


def make_demo(seed=21, n=600, shifted=False):
    """Create fictional customer-service data for a reproducible demo."""
    rng = np.random.default_rng(seed)
    age = rng.normal(36, 11, n).clip(18, 78)
    monthly_spend = rng.normal(72, 24, n).clip(5, 180)
    visits = rng.poisson(3.1, n)
    plan = rng.choice(["Basic", "Plus", "Premium"], n, p=[.52, .33, .15])
    region = rng.choice(["North", "South", "East", "West"], n, p=[.25, .25, .25, .25])
    satisfaction = rng.normal(7.2, 1.1, n).clip(1, 10)
    if shifted:
        monthly_spend = rng.normal(91, 31, n).clip(5, 220)
        visits = rng.poisson(4.0, n)
        plan = rng.choice(["Basic", "Plus", "Premium"], n, p=[.34, .42, .24])
        region = rng.choice(["North", "South", "East", "West"], n, p=[.18, .20, .38, .24])
        satisfaction = rng.normal(6.6, 1.5, n).clip(1, 10)
    df = pd.DataFrame({
        "age": age.round(0),
        "monthly_spend": monthly_spend.round(2),
        "visits_last_month": visits,
        "plan": plan,
        "region": region,
        "satisfaction_score": satisfaction.round(1),
    })
    # Introduce small missingness, slightly higher in the later sample.
    miss = .025 if not shifted else .07
    for col in ["monthly_spend", "satisfaction_score"]:
        mask = rng.random(n) < miss
        df.loc[mask, col] = np.nan
    return df


def read_csv(uploaded):
    return pd.read_csv(uploaded)


def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)


def psi_numeric(reference, current, bins=10):
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) < 5 or len(cur) < 5 or np.nanmin(ref) == np.nanmax(ref):
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-5, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-5, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(reference, current):
    ref = reference.fillna("⟨missing⟩").astype(str)
    cur = current.fillna("⟨missing⟩").astype(str)
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    if len(categories) > 100:
        return np.nan
    ref_pct = np.array([(ref == c).mean() for c in categories], dtype=float)
    cur_pct = np.array([(cur == c).mean() for c in categories], dtype=float)
    ref_pct = np.clip(ref_pct, 1e-5, None)
    cur_pct = np.clip(cur_pct, 1e-5, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compare_column(name, ref, cur, alpha):
    ref_nonmissing = ref.dropna()
    cur_nonmissing = cur.dropna()
    row = {
        "Feature": name,
        "Type": "Numeric" if is_numeric(ref) and is_numeric(cur) else "Categorical",
        "Reference rows": len(ref),
        "Current rows": len(cur),
        "Reference missing %": round(ref.isna().mean() * 100, 2),
        "Current missing %": round(cur.isna().mean() * 100, 2),
        "Missingness change (pp)": round((cur.isna().mean() - ref.isna().mean()) * 100, 2),
        "Reference unique": int(ref.nunique(dropna=True)),
        "Current unique": int(cur.nunique(dropna=True)),
    }
    p_value = np.nan
    if row["Type"] == "Numeric":
        a = pd.to_numeric(ref_nonmissing, errors="coerce").to_numpy()
        b = pd.to_numeric(cur_nonmissing, errors="coerce").to_numpy()
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        row["Reference mean"] = round(float(np.mean(a)), 4) if len(a) else np.nan
        row["Current mean"] = round(float(np.mean(b)), 4) if len(b) else np.nan
        row["Mean change %"] = round((np.mean(b) - np.mean(a)) / max(abs(np.mean(a)), 1e-9) * 100, 2) if len(a) and len(b) else np.nan
        row["KS statistic"] = np.nan
        if len(a) >= 2 and len(b) >= 2:
            test = ks_2samp(a, b)
            p_value = float(test.pvalue)
            row["KS statistic"] = round(float(test.statistic), 4)
        row["PSI"] = round(psi_numeric(ref, cur), 4) if np.isfinite(psi_numeric(ref, cur)) else np.nan
    else:
        a = ref.fillna("⟨missing⟩").astype(str)
        b = cur.fillna("⟨missing⟩").astype(str)
        row["Reference mean"] = np.nan
        row["Current mean"] = np.nan
        row["Mean change %"] = np.nan
        row["KS statistic"] = np.nan
        row["PSI"] = round(psi_categorical(ref, cur), 4) if np.isfinite(psi_categorical(ref, cur)) else np.nan
        try:
            levels = sorted(set(a.unique()) | set(b.unique()))
            table = pd.DataFrame({
                "reference": [int((a == level).sum()) for level in levels],
                "current": [int((b == level).sum()) for level in levels],
            }, index=levels)
            table = table.loc[table.sum(axis=1) > 0]
            if len(table) >= 2 and table.shape[1] == 2:
                p_value = float(chi2_contingency(table.to_numpy())[1])
        except (ValueError, ZeroDivisionError):
            p_value = np.nan
    row["Test p-value"] = round(p_value, 6) if np.isfinite(p_value) else np.nan
    row["Statistically changed"] = bool(p_value < alpha) if np.isfinite(p_value) else False
    psi = row.get("PSI", np.nan)
    # PSI thresholds are practical heuristics, not universal statistical rules.
    row["PSI flag"] = "High" if np.isfinite(psi) and psi >= .25 else ("Moderate" if np.isfinite(psi) and psi >= .10 else "Low/unknown")
    row["Interpretation"] = make_interpretation(row)
    return row


def make_interpretation(row):
    messages = []
    if abs(row["Missingness change (pp)"]) >= 3:
        messages.append(f"missing values changed by {row['Missingness change (pp)']:+.1f} percentage points")
    psi = row.get("PSI", np.nan)
    if np.isfinite(psi) and psi >= .25:
        messages.append("large distribution shift by PSI heuristic")
    elif np.isfinite(psi) and psi >= .10:
        messages.append("moderate distribution shift by PSI heuristic")
    if row["Statistically changed"]:
        messages.append("two-sample test detected a difference at the selected significance level")
    if not messages:
        messages.append("no strong shift flagged by the selected checks")
    return "; ".join(messages).capitalize() + "."


st.sidebar.markdown("## 🔎 ShiftScope")
st.sidebar.caption("Understand what changed between two datasets.")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Data source", ["Use built-in demo", "Upload two CSV files"])
alpha = st.sidebar.select_slider("Significance level", options=[0.01, 0.05, 0.10], value=0.05)
st.sidebar.caption("Statistical significance can be affected by sample size. Combine tests with domain knowledge.")

if mode == "Use built-in demo":
    reference = make_demo(21, shifted=False)
    current = make_demo(42, shifted=True)
    st.sidebar.success("Loaded fictional reference and current datasets.")
else:
    ref_file = st.sidebar.file_uploader("Reference / baseline CSV", type=["csv"], key="ref")
    cur_file = st.sidebar.file_uploader("Current / new CSV", type=["csv"], key="cur")
    if ref_file is None or cur_file is None:
        st.markdown("""
        <div class="hero"><h1>ShiftScope 🔎</h1>
        <p>Compare two datasets and discover distribution changes, missing-data shifts, and potential data-quality issues.</p></div>
        """, unsafe_allow_html=True)
        st.info("Upload both CSV files in the sidebar to start, or choose **Use built-in demo**.")
        st.stop()
    try:
        reference = read_csv(ref_file)
        current = read_csv(cur_file)
    except Exception as exc:
        st.error(f"Could not read one of the CSV files: {exc}")
        st.stop()

st.markdown("""
<div class="hero">
  <h1>ShiftScope 🔎</h1>
  <p>Explainable data drift auditing — see what changed, how much it changed, and what to investigate next.</p>
</div>
""", unsafe_allow_html=True)

common = [c for c in reference.columns if c in current.columns]
only_ref = [c for c in reference.columns if c not in current.columns]
only_cur = [c for c in current.columns if c not in reference.columns]
if not common:
    st.error("The datasets have no columns in common. Please upload files with matching feature names.")
    st.stop()

chosen = st.multiselect("Features to compare", common, default=common)
if not chosen:
    st.warning("Select at least one feature.")
    st.stop()

ref = reference[chosen].copy()
cur = current[chosen].copy()
# Treat a feature as numeric only when both versions are numeric; otherwise compare as categories.
for col in chosen:
    if is_numeric(ref[col]) and is_numeric(cur[col]):
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
        cur[col] = pd.to_numeric(cur[col], errors="coerce")
    else:
        ref[col] = ref[col].astype("string")
        cur[col] = cur[col].astype("string")

results = pd.DataFrame([compare_column(c, ref[c], cur[c], alpha) for c in chosen])
changed = int(results["Statistically changed"].sum())
high_psi = int((results["PSI flag"] == "High").sum())
missing_shift = int((results["Missingness change (pp)"].abs() >= 3).sum())

k1, k2, k3, k4 = st.columns(4)
for col, label, value, note in [
    (k1, "REFERENCE ROWS", f"{len(reference):,}", "Baseline dataset"),
    (k2, "CURRENT ROWS", f"{len(current):,}", "New dataset"),
    (k3, "FEATURES COMPARED", str(len(chosen)), "Shared selected columns"),
    (k4, "SHIFT SIGNALS", str(changed + high_psi), "Test and PSI flags; may overlap"),
]:
    with col:
        st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)

if only_ref or only_cur:
    st.warning(f"Schema difference — only in reference: {only_ref or 'none'}; only in current: {only_cur or 'none'}.")

st.write("")
left, right = st.columns([1.2, 1])
with left:
    st.subheader("Feature-by-feature audit")
    st.dataframe(results, use_container_width=True, hide_index=True)
with right:
    st.subheader("Distribution shift scores")
    chart_data = results.dropna(subset=["PSI"]).sort_values("PSI", ascending=True)
    if chart_data.empty:
        st.info("PSI could not be calculated for the selected features.")
    else:
        fig = px.bar(chart_data, x="PSI", y="Feature", orientation="h", color="PSI",
                     color_continuous_scale=["#4a9b83", "#e7b45b", "#c85b62"])
        fig.add_vline(x=.10, line_dash="dash", line_color="#b98534", annotation_text="0.10")
        fig.add_vline(x=.25, line_dash="dash", line_color="#b84b54", annotation_text="0.25")
        fig.update_layout(height=max(300, 42 * len(chart_data)), margin=dict(l=5, r=5, t=15, b=5),
                          coloraxis_showscale=False, xaxis_title="Population Stability Index (PSI)",
                          yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Visual comparison")
feature = st.selectbox("Choose a feature to inspect", chosen)
is_num = is_numeric(ref[feature]) and is_numeric(cur[feature])
plot_ref = pd.DataFrame({"Value": ref[feature].dropna(), "Dataset": "Reference"})
plot_cur = pd.DataFrame({"Value": cur[feature].dropna(), "Dataset": "Current"})
plot_df = pd.concat([plot_ref, plot_cur], ignore_index=True)
if is_num:
    fig = px.histogram(plot_df, x="Value", color="Dataset", barmode="overlay", histnorm="probability density",
                       opacity=.58, color_discrete_map={"Reference": "#315c9b", "Current": "#3b9b83"})
    fig.update_layout(yaxis_title="Density", bargap=.06)
else:
    counts = plot_df.groupby(["Dataset", "Value"], dropna=False).size().reset_index(name="Count")
    counts["Share (%)"] = counts.groupby("Dataset")["Count"].transform(lambda s: s / s.sum() * 100)
    fig = px.bar(counts, x="Value", y="Share (%)", color="Dataset", barmode="group",
                 color_discrete_map={"Reference": "#315c9b", "Current": "#3b9b83"})
fig.update_layout(height=350, margin=dict(l=5, r=5, t=15, b=5),
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("What should I investigate?")
for _, row in results.sort_values(["Statistically changed", "PSI"], ascending=False, na_position="last").iterrows():
    if row["Statistically changed"] or row["PSI flag"] in ["Moderate", "High"] or abs(row["Missingness change (pp)"]) >= 3:
        st.markdown(f"- **{row['Feature']}** — {row['Interpretation']} PSI: `{row['PSI']}`; p-value: `{row['Test p-value']}`.")
if not results["Statistically changed"].any() and not (results["PSI flag"].isin(["Moderate", "High"])).any() and not missing_shift:
    st.success("No strong changes were flagged by these checks. This does not prove the datasets are identical.")

st.caption("Method notes: numeric features use the two-sample Kolmogorov–Smirnov test; categorical features use a chi-square test when feasible. PSI flags use common heuristic thresholds (0.10 and 0.25), not universal rules. Multiple-testing correction is not applied. This tool supports investigation and is not a substitute for domain review.")

csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download audit report (CSV)", data=csv_bytes, file_name="shiftscope_audit_report.csv", mime="text/csv")
