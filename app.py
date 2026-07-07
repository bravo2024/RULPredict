"""app.py – RULPredict: comprehensive predictive maintenance & RUL forecasting dashboard."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats

from src.data import make_synthetic, fetch_ai4i
from src.model import fit_and_evaluate, forecast_next

st.set_page_config(page_title="RULPredict | Predictive Maintenance",
                   layout="wide", page_icon="⚙")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙ Configuration")
    data_source = st.radio("Dataset",
        ["AI4I 2020 Predictive Maintenance (live)", "Synthetic (demo)", "Upload CSV"], index=0)
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload health-index CSV", type=["csv"])
    horizon = st.slider("Forecast horizon (cycles)", 15, 120, 60, step=5)
    threshold = st.slider("Maintenance threshold (health index)", 5.0, 60.0, 30.0, step=1.0)
    season_period = st.slider("Cycle period (for decomposition)", 20, 200, 100, step=10)
    n_folds = st.slider("Walk-forward folds", 3, 8, 5, step=1)
    seed = st.number_input("Synthetic seed", min_value=1, max_value=9999, value=42, step=1)
    n_cycles = st.slider("Synthetic cycles", 120, 500, 240, step=20)
    st.divider()
    st.caption("AI4I 2020 Predictive Maintenance — UCI ML Repository #601. Tool wear → health index.")
    st.code("streamlit run app.py", language="bash")

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching AI4I 2020 data...")
def _load(source: str, n: int, sd: int):
    if "AI4I" in source:
        try:
            r = fetch_ai4i()
            return r["series"], r["source"], "Health index (0–100)", "cycles"
        except Exception:
            pass
    s = make_synthetic(n=n, seed=sd)
    return s["series"], "Synthetic health-index signal", "Health index", "cycles"

if data_source == "Upload CSV" and uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    num_cols = uploaded_df.select_dtypes("number").columns.tolist()
    if not num_cols:
        st.error("CSV must contain at least one numeric column."); st.stop()
    value_col = st.sidebar.selectbox("Health-index column", num_cols)
    series = uploaded_df[value_col].dropna().astype(float).to_numpy()
    source_label = f"Uploaded: {uploaded_file.name} / {value_col}"
    y_label, freq_label = "Health index", "cycles"
else:
    series, source_label, y_label, freq_label = _load(data_source, n_cycles, seed)

if len(series) < 40:
    st.error("Need at least 40 observations."); st.stop()

series_key = tuple(float(x) for x in series)

# ── Train & helpers ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training ridge model...")
def _train(key: tuple):
    return fit_and_evaluate({"series": np.asarray(key, float)})

@st.cache_data(show_spinner="Computing test residuals...")
def _residuals(key: tuple, n: int = 80):
    s = np.asarray(key, float)
    if len(s) < 35:
        return np.zeros(1)
    m, _ = fit_and_evaluate({"series": s})
    start = max(29, len(s) - n)
    out = []
    for t in range(start, len(s)):
        try:
            out.append(float(s[t]) - float(forecast_next(m, list(s[:t]))))
        except Exception:
            pass
    return np.asarray(out) if out else np.zeros(1)

@st.cache_data(show_spinner="Generating forecast...")
def _forecast(key: tuple, h: int):
    m, _ = fit_and_evaluate({"series": np.asarray(key, float)})
    tail = list(np.asarray(key, float))
    out = []
    for _ in range(h):
        v = float(np.clip(forecast_next(m, tail), 0.0, 100.0))
        out.append(v); tail.append(v)
    return np.asarray(out)

model, metrics = _train(series_key)
walk_resid = _residuals(series_key)
preds = _forecast(series_key, horizon)

# ── RUL estimation ─────────────────────────────────────────────────────────────
def _rul(s: np.ndarray, fc: np.ndarray, thresh: float):
    if len(s) and s[-1] <= thresh:
        return 0
    hits = np.where(fc <= thresh)[0]
    return int(hits[0] + 1) if len(hits) else None

rul_cycles = _rul(series, preds, threshold)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("⚙ RULPredict")
st.caption(f"{source_label}  |  {len(series):,} observations  |  frequency: {freq_label}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Observed Cycles", f"{len(series):,}")
c2.metric("Current Health", f"{series[-1]:.1f}")
c3.metric("RMSE", f"{metrics['rmse']:.4f}")
c4.metric("SMAPE", f"{metrics['smape_pct']:.2f}%")
c5.metric("Estimated RUL",
          f"{rul_cycles} cycles" if rul_cycles is not None and rul_cycles > 0
          else (f">{horizon} cycles" if rul_cycles is None else "FAILED"))

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Explorer", "📐 Stationarity & ACF",
    "🌀 Decomposition", "🔄 Walk-Forward CV", "🔮 Forecast & RUL"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data Explorer
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Health-Index Series & Descriptive Statistics")
    st.markdown("""
**Health index** = 100 × (1 − tool_wear / 253), clipped to [0, 100].
Tool wear increases monotonically from 0 → ~253 min over 10,000 machine cycles.
Health starts near 100 (new tool) and degrades toward 0 (failure threshold).
""")

    win = min(50, max(5, len(series) // 10))
    rm = np.convolve(series, np.ones(win) / win, mode="valid")
    rs = np.array([series[i:i + win].std(ddof=1) for i in range(len(series) - win + 1)])
    xr = np.arange(win - 1, len(series))

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    axes[0].plot(series, color="#1d4ed8", lw=0.5, alpha=0.8, label="Health index")
    axes[0].plot(xr, rm, color="#dc2626", lw=1.5, label=f"{win}-pt rolling mean")
    axes[0].axhline(threshold, color="#f59e0b", lw=1.2, ls="-.", label=f"Maintenance threshold ({threshold:.0f})")
    axes[0].set_ylabel(y_label); axes[0].legend(fontsize=8)
    axes[0].set_title("Tool health degradation over machine cycles")
    axes[0].set_ylim(-5, 105)
    axes[1].plot(xr, rs, color="#7c3aed", lw=1.2, label=f"{win}-pt rolling std (ddof=1)")
    axes[1].set_xlabel(f"Time ({freq_label})"); axes[1].set_ylabel("Std dev"); axes[1].legend(fontsize=8)
    axes[1].set_title("Rolling std — spikes indicate rapid degradation events")
    fig.tight_layout(); st.pyplot(fig, use_container_width=True)

    col_s, col_h = st.columns(2)
    with col_s:
        st.markdown("**Descriptive Statistics**")
        sk = float(stats.skew(series)); ku = float(stats.kurtosis(series))
        cv = series.std(ddof=1) / abs(series.mean()) * 100 if series.mean() != 0 else float("nan")
        recent_slope = float(np.polyfit(np.arange(min(50, len(series))),
                                        series[-min(50, len(series)):], 1)[0])
        st.dataframe(pd.DataFrame({
            "Statistic": ["N", "Mean", "Median", "Std (σ, ddof=1)", "Min", "Max",
                          "Recent slope (last 50)", "Skewness", "Excess Kurtosis", "CV (%)"],
            "Value": [f"{len(series):,}", f"{series.mean():.4f}", f"{np.median(series):.4f}",
                      f"{series.std(ddof=1):.4f}", f"{series.min():.4f}", f"{series.max():.4f}",
                      f"{recent_slope:.4f}/cycle", f"{sk:.4f}", f"{ku:.4f}", f"{cv:.2f}%"]
        }), use_container_width=True, hide_index=True)
        st.markdown("""
**Recent slope** < 0 → active degradation; steeper = faster failure approach.
**CV** = σ/|μ| × 100 — rises near end-of-life as health fluctuates more.
**Bessel's correction** (ddof=1): unbiased sample standard deviation.
""")
    with col_h:
        st.markdown("**Distribution Histogram**")
        fig_h, ax_h = plt.subplots(figsize=(6, 4))
        ax_h.hist(series, bins=50, color="#1d4ed8", alpha=0.7, edgecolor="white")
        ax_h.axvline(series.mean(), color="#dc2626", ls="--", lw=1.5, label=f"Mean={series.mean():.2f}")
        ax_h.axvline(np.median(series), color="#f59e0b", ls="--", lw=1.5,
                     label=f"Median={np.median(series):.2f}")
        ax_h.axvline(threshold, color="#7c3aed", ls="-.", lw=1.5, label=f"Threshold={threshold:.0f}")
        ax_h.set_xlabel(y_label); ax_h.set_ylabel("Count"); ax_h.legend(fontsize=8)
        fig_h.tight_layout(); st.pyplot(fig_h, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Stationarity & ACF/PACF
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Stationarity Test & Autocorrelation Structure")

    try:
        from statsmodels.tsa.stattools import adfuller, acf as sm_acf, pacf as sm_pacf
        adf = adfuller(series, autolag="AIC")
        adf_stat, adf_p, adf_lag, adf_n, adf_crit = adf[0], adf[1], adf[2], adf[3], adf[4]
        sm_ok = True
    except Exception as exc:
        st.error(f"statsmodels unavailable: {exc}"); sm_ok = False

    if sm_ok:
        st.markdown("#### Augmented Dickey-Fuller (ADF) Unit Root Test")
        st.latex(r"\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{k}\delta_i\,\Delta y_{t-i} + \varepsilon_t")
        st.markdown(r"""
**H₀**: γ = 0 — unit root, series is **non-stationary**.
**H₁**: γ < 0 — series is **stationary**.
Health-index degradation is typically **trend-stationary** (deterministic downward trend).
For RUL modelling, the trend IS the primary signal — we preserve it rather than differencing.
""")
        col1, col2 = st.columns(2)
        with col1:
            verdict = "✅ Stationary — reject H₀" if adf_p < 0.05 else "❌ Non-stationary — fail to reject H₀"
            st.dataframe(pd.DataFrame({
                "Metric": ["ADF Statistic", "p-value", "Lags used (AIC)", "Obs in test", "Verdict"],
                "Value": [f"{adf_stat:.4f}", f"{adf_p:.6f}", str(adf_lag), str(adf_n), verdict]
            }), use_container_width=True, hide_index=True)
        with col2:
            st.dataframe(pd.DataFrame({
                "Confidence": ["1% (MacKinnon)", "5% (MacKinnon)", "10% (MacKinnon)"],
                "Critical Value": [f"{adf_crit['1%']:.4f}", f"{adf_crit['5%']:.4f}",
                                   f"{adf_crit['10%']:.4f}"],
                "ADF < CV?": ["✅ Reject H₀" if adf_stat < adf_crit[k] else "❌ No"
                              for k in ["1%", "5%", "10%"]]
            }), use_container_width=True, hide_index=True)
        st.info("For degradation series: non-stationarity is expected — the downward trend captures RUL. "
                "Differencing would destroy this signal. Lag-feature ridge implicitly models trend via AR structure.")

        st.markdown("---")
        st.markdown("#### ACF & PACF — Autocorrelation Structure")
        st.markdown(r"""
**ACF(k)** = Corr(yₜ, yₜ₋ₖ) — total autocorrelation; slow decay → strong trend / non-stationarity.
**PACF(k)** — partial correlation at lag k, controlling for 1…k-1.
Confidence bands: **±1.96/√n** (Bartlett). Red bars are significant at α = 0.05.
""")
        nlags = min(60, len(series) // 4)
        acf_v = sm_acf(series, nlags=nlags, fft=True)
        pacf_v = sm_pacf(series, nlags=nlags, method="ywm")
        conf = 1.96 / np.sqrt(len(series))
        lx = np.arange(len(acf_v))

        fig_a, ax_a = plt.subplots(1, 2, figsize=(14, 4))
        for ax, vals, title, ylabel in [
            (ax_a[0], acf_v, "ACF — slow decay confirms trend / non-stationarity", "ACF"),
            (ax_a[1], pacf_v, "PACF — cut-off at lag 1–2 → AR structure in differences", "PACF"),
        ]:
            colors = ["#dc2626" if abs(v) > conf else "#93c5fd" for v in vals]
            ax.bar(lx, vals, color=colors, width=0.6)
            ax.axhline(conf, ls="--", color="gray", lw=0.8, label=f"±{conf:.3f}")
            ax.axhline(-conf, ls="--", color="gray", lw=0.8)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xlabel("Lag (cycles)"); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.legend(fontsize=7)
        fig_a.tight_layout(); st.pyplot(fig_a, use_container_width=True)

        st.markdown("""
**Degradation series interpretation:**
- Slow positive ACF decay across all lags → strong trend (I(1) or trend-stationary)
- PACF cuts off at 1–2 → AR(1) or AR(2) dominates after detrending
- No seasonal spikes unless scheduled maintenance creates periodicity

| Pattern | Model |
|---|---|
| Slow ACF decay, PACF cuts off at 1 | AR(1) on first differences |
| Both tail off after differencing | ARIMA(1,1,1) |
| Our model | Lag-feature ridge: AR structure up to lag 28 |
""")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Decomposition
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Seasonal Decomposition & Frequency Analysis")
    st.markdown(r"""
**Additive decomposition:** $y_t = T_t + S_t + R_t$

For health-index data:
- **T** = degradation trend (the primary signal for RUL)
- **S** = periodic maintenance/replacement cycle (if any)
- **R** = measurement noise and abrupt events

The trend component directly visualises the degradation trajectory that drives RUL estimation.
""")

    if len(series) < 2 * season_period + 1:
        st.warning(f"Series too short for period={season_period}. Need ≥ {2 * season_period + 1} cycles.")
    else:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            dec = seasonal_decompose(series, model="additive", period=season_period,
                                     extrapolate_trend="freq")
            T, S, R = dec.trend, dec.seasonal, dec.resid

            var_r = np.nanvar(R)
            fs = max(0.0, 1 - var_r / np.nanvar(S + R)) if np.nanvar(S + R) > 0 else 0.0
            ft = max(0.0, 1 - var_r / np.nanvar(T + R)) if np.nanvar(T + R) > 0 else 0.0

            fig_d, axd = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
            for ax, vals, lbl, col, lw in [
                (axd[0], series, "Observed (health index)", "#1d4ed8", 0.5),
                (axd[1], T, "Trend (T) — degradation signal", "#dc2626", 1.5),
                (axd[2], S, f"Seasonal (S, period={season_period} cycles)", "#16a34a", 1.0),
                (axd[3], R, "Residual (R) — noise + shocks", "#7c3aed", 0.7),
            ]:
                ax.plot(vals, color=col, lw=lw); ax.set_ylabel(lbl)
                if lbl != "Observed (health index)":
                    ax.axhline(0, color="gray", lw=0.5, ls="--")
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            axd[0].axhline(threshold, color="#f59e0b", lw=1.0, ls="-.",
                           label=f"Threshold={threshold:.0f}")
            axd[0].legend(fontsize=7); axd[0].set_ylim(-5, 105)
            axd[3].set_xlabel(f"Time ({freq_label})")
            fig_d.suptitle(f"Additive Decomposition — period={season_period} cycles", y=1.01)
            fig_d.tight_layout(); st.pyplot(fig_d, use_container_width=True)

            mc1, mc2 = st.columns(2)
            mc1.metric("Seasonality Strength Fₛ", f"{fs:.4f}",
                       help="1 - Var(R)/Var(S+R). > 0.64 = strong periodic pattern (Wang et al. 2006)")
            mc2.metric("Trend Strength F_T", f"{ft:.4f}",
                       help="1 - Var(R)/Var(T+R). > 0.64 = dominant degradation trend")
        except Exception as exc:
            st.error(f"Decomposition failed: {exc}")

        st.markdown("---")
        st.markdown("#### Periodogram — Power Spectral Density via FFT")
        st.markdown(r"""
$$I(\omega_j) = \frac{1}{n}\left|\sum_{t=0}^{n-1} y_t\, e^{-2\pi i\, \omega_j t}\right|^2$$

For degradation: dominant power at very low frequencies (long periods) due to the downward trend.
Peaks at specific cycle lengths indicate periodic maintenance or batch effects.
""")
        fv = np.fft.rfft(series - series.mean())
        fr = np.fft.rfftfreq(len(series))[1:]
        pw = (np.abs(fv) ** 2)[1:]
        per = 1.0 / fr

        fig_fft, ax_fft = plt.subplots(figsize=(13, 3))
        ax_fft.plot(per, pw, color="#1d4ed8", lw=0.8)
        ax_fft.set_xscale("log")
        ax_fft.set_xlabel(f"Period ({freq_label}/cycle) — log scale")
        ax_fft.set_ylabel("Power")
        ax_fft.set_title("Periodogram — low-frequency dominance is expected for degradation trend")
        top_idx = np.argsort(pw)[-5:][::-1]
        for i in top_idx:
            ax_fft.axvline(per[i], color="#dc2626", lw=1.0, ls="--", alpha=0.5)
            ax_fft.text(per[i], pw[i] * 0.85, f" T≈{per[i]:.0f}", fontsize=7, color="#dc2626")
        fig_fft.tight_layout(); st.pyplot(fig_fft, use_container_width=True)

        st.dataframe(
            pd.DataFrame([(f"{per[i]:.0f} {freq_label}", f"{pw[i]:.3e}") for i in top_idx[:5]],
                         columns=["Dominant Period", "Power"]),
            use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Walk-Forward CV
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Walk-Forward (Expanding-Window) Cross-Validation")
    st.markdown(r"""
**Walk-forward** is the correct CV for time series — k-fold shuffling leaks future health
states into training, giving optimistically biased RMSE.

For degradation: earlier folds (early-life, high health) may show different errors than later folds
(end-of-life, rapid degradation). Monitoring RMSE across folds reveals regime dependence.

$$\text{RMSE}_{\text{fold}} = \sqrt{\frac{1}{n_{\text{test}}}\sum_t(\hat{y}_t - y_t)^2}$$
**Lag leakage prevention:** gap = max(lags) = 28 cycles between train and test boundary.
""")

    min_size = max(100, 29 * 3)
    if len(series) < min_size + 20:
        st.warning(f"Series too short for walk-forward CV (need ≥ {min_size + 20} obs).")
    else:
        with st.spinner("Running expanding-window cross-validation…"):
            incr = max(1, (len(series) - min_size) // n_folds)
            rows = []
            for fold in range(n_folds):
                fe = min(len(series), min_size + (fold + 1) * incr)
                try:
                    _, fm = fit_and_evaluate({"series": series[:fe]})
                    rows.append({
                        "Fold": fold + 1, "Train size": fe,
                        "RMSE": round(fm["rmse"], 4),
                        "MAE": round(fm.get("mae", float("nan")), 4),
                        "SMAPE (%)": round(fm["smape_pct"], 4),
                    })
                except Exception:
                    rows.append({"Fold": fold + 1, "Train size": fe,
                                 "RMSE": float("nan"), "MAE": float("nan"), "SMAPE (%)": float("nan")})

        fold_df = pd.DataFrame(rows)
        st.dataframe(fold_df, use_container_width=True, hide_index=True)

        fig_wf, axw = plt.subplots(1, 2, figsize=(12, 4))
        fols = fold_df["Fold"]
        axw[0].bar(fols, fold_df["RMSE"], color="#1d4ed8", alpha=0.8)
        axw[0].set_xlabel("Fold"); axw[0].set_ylabel("RMSE (health index units)")
        axw[0].set_title("RMSE by fold (expanding training window)"); axw[0].set_xticks(fols)
        axw[1].bar(fols, fold_df["SMAPE (%)"], color="#dc2626", alpha=0.8)
        axw[1].set_xlabel("Fold"); axw[1].set_ylabel("SMAPE (%)")
        axw[1].set_title("SMAPE by fold"); axw[1].set_xticks(fols)
        fig_wf.tight_layout(); st.pyplot(fig_wf, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Ljung-Box Q-Test — Residual White Noise Check")
        st.markdown(r"""
**H₀**: residuals are white noise (model has extracted all autocorrelation).

$$Q_m = n(n+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{n-k} \;\overset{H_0}{\sim}\; \chi^2(m)$$

p < 0.05 → residual autocorrelation remains → model has not fully captured the trend or cycle structure.
""")
        if len(walk_resid) > 10:
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb = acorr_ljungbox(walk_resid, lags=[10, 20, 30], return_df=True)
                lb.index = [10, 20, 30]; lb.columns = ["Q-statistic", "p-value"]
                st.dataframe(lb.round(4), use_container_width=True)
                if (lb["p-value"] < 0.05).any():
                    st.warning("p < 0.05 → residual autocorrelation; trend not fully captured — consider more lags.")
                else:
                    st.success("All p ≥ 0.05 → residuals consistent with white noise. ✓")
            except Exception as exc:
                st.error(f"Ljung-Box failed: {exc}")
        else:
            st.info("Too few residuals for Ljung-Box test.")

        st.markdown("---")
        st.markdown("#### Residual Diagnostics")
        if len(walk_resid) > 5:
            fig_r, axr = plt.subplots(1, 2, figsize=(12, 4))
            axr[0].hist(walk_resid, bins=min(30, max(5, len(walk_resid) // 3)),
                        color="#7c3aed", alpha=0.7, edgecolor="white")
            axr[0].axvline(0, color="black", lw=1.0, ls="--")
            axr[0].axvline(walk_resid.mean(), color="#dc2626", lw=1.2, ls="--",
                           label=f"Mean={walk_resid.mean():.4f}")
            axr[0].set_xlabel("Residual (health index)"); axr[0].set_ylabel("Count")
            axr[0].legend(fontsize=8)
            axr[0].set_title("Residual distribution — centred at 0 = unbiased forecast")
            axr[1].scatter(np.arange(len(walk_resid)), walk_resid, s=8, color="#7c3aed", alpha=0.6)
            axr[1].axhline(0, color="black", lw=1.0, ls="--")
            axr[1].set_xlabel("Test observation index"); axr[1].set_ylabel("Residual")
            axr[1].set_title("Residuals vs time — no pattern = well-specified model")
            fig_r.tight_layout(); st.pyplot(fig_r, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Forecast & RUL
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader(f"{horizon}-Cycle RUL Forecast + Bootstrap Prediction Intervals (90%)")
    st.markdown(r"""
**Bootstrap PI for RUL forecasting:**
1. Collect signed walk-forward residuals: $e_t = y_t - \hat{y}_t$
2. Draw *B = 500* paths of length *h* with replacement (preserves empirical error distribution)
3. Add each path to the recursive point forecast → 500 health trajectories
4. 90% PI = [5th, 95th] percentile at each step; RUL from each trajectory → bootstrap RUL distribution

Health clipped to [0, 100] — cannot exceed full health or go negative.
""")

    rng = np.random.default_rng(42)
    resid_pool = walk_resid if len(walk_resid) >= 5 else rng.normal(0, metrics["rmse"], 200)
    n_boot = 500
    boot = rng.choice(resid_pool, size=(n_boot, horizon), replace=True)
    boot_preds = np.clip(preds[np.newaxis, :] + boot, 0.0, 100.0)
    lo90 = np.percentile(boot_preds, 5, axis=0)
    hi90 = np.percentile(boot_preds, 95, axis=0)

    boot_rul = []
    for b in range(n_boot):
        hits = np.where(boot_preds[b] <= threshold)[0]
        boot_rul.append(int(hits[0] + 1) if len(hits) else horizon + 1)
    boot_rul = np.array(boot_rul)

    hist_n = min(500, len(series))
    x_h = np.arange(len(series) - hist_n, len(series))
    x_f = np.arange(len(series), len(series) + horizon)

    fig_fc, ax_fc = plt.subplots(figsize=(13, 4))
    ax_fc.plot(x_h, series[-hist_n:], color="#1d4ed8", lw=0.8,
               label=f"Observed (last {hist_n} cycles)")
    ax_fc.plot(x_f, preds, color="#dc2626", lw=2.0, ls="--", label="Point forecast")
    ax_fc.fill_between(x_f, lo90, hi90, color="#dc2626", alpha=0.2, label="90% Bootstrap PI")
    ax_fc.axhline(threshold, color="#f59e0b", lw=1.5, ls="-.",
                  label=f"Maintenance threshold ({threshold:.0f})")
    ax_fc.axvline(len(series) - 1, color="gray", lw=0.8, ls=":")
    ax_fc.set_xlabel(f"Time ({freq_label})"); ax_fc.set_ylabel(y_label)
    ax_fc.legend(loc="upper right", fontsize=8); ax_fc.set_ylim(-5, 105)
    rul_str = f">{horizon}" if rul_cycles is None else str(rul_cycles)
    ax_fc.set_title(f"Health-index forecast — RUL estimate: {rul_str} cycles")
    fig_fc.tight_layout(); st.pyplot(fig_fc, use_container_width=True)

    st.markdown("---")
    st.markdown("#### RUL Uncertainty — Bootstrap Distribution")
    col_rul1, col_rul2 = st.columns(2)
    with col_rul1:
        fig_rul, ax_rul = plt.subplots(figsize=(6, 3))
        ax_rul.hist(boot_rul, bins=min(30, horizon // 2 + 1),
                    color="#7c3aed", alpha=0.7, edgecolor="white")
        ax_rul.axvline(np.median(boot_rul), color="#dc2626", lw=1.5, ls="--",
                       label=f"Median RUL={np.median(boot_rul):.0f}")
        ax_rul.set_xlabel("Bootstrap RUL (cycles)"); ax_rul.set_ylabel("Count")
        ax_rul.set_title("Bootstrap RUL distribution (500 trajectories)")
        ax_rul.legend(fontsize=8)
        fig_rul.tight_layout(); st.pyplot(fig_rul, use_container_width=True)
    with col_rul2:
        st.markdown("**RUL Summary**")
        st.dataframe(pd.DataFrame({
            "Metric": ["Point estimate RUL", "Bootstrap 5th pct (pessimistic)",
                       "Bootstrap median RUL", "Bootstrap 95th pct (optimistic)",
                       "P(failure within horizon)", "Current health", "Threshold"],
            "Value": [
                f"{rul_cycles} cycles" if rul_cycles else f">{horizon} cycles",
                f"{int(np.percentile(boot_rul, 5))} cycles",
                f"{int(np.median(boot_rul))} cycles",
                f"{int(np.percentile(boot_rul, 95))} cycles",
                f"{100 * (boot_rul <= horizon).mean():.1f}%",
                f"{series[-1]:.2f}",
                f"{threshold:.1f}",
            ]
        }), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Horizon Decay — PI Width by Step Ahead")
    pi_width = hi90 - lo90
    fig_hd, ax_hd = plt.subplots(figsize=(10, 3))
    ax_hd.plot(np.arange(1, horizon + 1), pi_width, "o-", color="#7c3aed", ms=4, lw=1.2)
    ax_hd.set_xlabel("Step ahead (h)"); ax_hd.set_ylabel("90% PI width (health index units)")
    ax_hd.set_title("Horizon decay — uncertainty grows as recursive forecast substitutes predicted values for actuals")
    ax_hd.grid(True, alpha=0.3)
    fig_hd.tight_layout(); st.pyplot(fig_hd, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Maintenance Decision")
    recent_slope = float(np.polyfit(np.arange(min(20, len(series))),
                                    series[-min(20, len(series)):], 1)[0])
    if series[-1] <= threshold:
        st.error(f"Current health ({series[-1]:.2f}) ≤ threshold ({threshold:.1f}). Immediate inspection required.")
    elif rul_cycles is not None and rul_cycles <= 15:
        st.error(f"Threshold breach in ~{rul_cycles} cycles. Schedule inspection immediately.")
    elif rul_cycles is not None and rul_cycles <= 45:
        st.warning(f"Threshold breach in ~{rul_cycles} cycles. Add to next maintenance window.")
    elif rul_cycles is not None:
        st.info(f"Threshold breach in ~{rul_cycles} cycles. Monitor trend; not immediate.")
    else:
        st.success(f"No threshold breach within {horizon}-cycle forecast horizon. Continue monitoring.")

    dcols = st.columns(3)
    dcols[0].metric("Current Health", f"{series[-1]:.2f}")
    dcols[1].metric("Recent Slope (last 20)", f"{recent_slope:.4f}/cycle")
    dcols[2].metric("Estimated RUL", f"{rul_str} cycles")

    st.markdown("---")
    fc_df = pd.DataFrame({
        "Cycle (h)": np.arange(1, horizon + 1),
        "Health Forecast": np.round(preds, 4),
        "Lower 90% PI": np.round(lo90, 4),
        "Upper 90% PI": np.round(hi90, 4),
        "Below Threshold": preds <= threshold,
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download forecast CSV",
                       fc_df.to_csv(index=False).encode(),
                       "rulpredict_forecast.csv", "text/csv")

    st.markdown("---")
    st.markdown(f"""
**Model:** Ridge Regression on lag features
**Health index:** 100 × (1 − tool_wear / 253) clipped to [0, 100] (AI4I 2020 — UCI #601)
**Features:** lags [1, 2, 3, 7, 14, 28] + sin/cos cyclic encoding (period={season_period} cycles)
**Split:** chronological 80/20 with gap = 28 cycles (prevents lag-feature data leakage)
**Ridge:** β = (X'X + αI)⁻¹ X'y — L2 regularisation, intercept excluded from penalty
**RUL:** first forecast step where health ≤ {threshold:.1f}
""")
    st.json({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
