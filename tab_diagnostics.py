"""
tab_diagnostics.py — Regression diagnostics tab
Residual tests, heteroscedasticity, autocorrelation, normality
"""
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, txt_s, p, steps_html, two_col, three_col,
    table_html, metric_row, section_heading, S, FH, FB, FM, TXT, NO_SEL
)


def tab_diagnostics():
    # ── 1. Diagnostic Overview ───────────────────────────────────
    render_card("🔬 Regression Diagnostics — Testing CLRM Assumptions",
        p(f'Before trusting regression outputs, verify each CLRM assumption. '
          f'{rt2("Violations invalidate")} inference (t-tests, F-tests, confidence intervals).')
        + three_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">🧪 Normality of Residuals</span><br>'
               + p(f'{bdg("Jarque-Bera","gold")} Tests skewness + kurtosis<br>'
                   f'{bdg("Shapiro-Wilk","blue")} More powerful for n<50<br>'
                   f'{bdg("Q-Q Plot","purple")} Visual check')
               + p(f'H₀: residuals are normally distributed'),
               "gold"),
            ib(f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545;font-weight:600">📈 Heteroscedasticity</span><br>'
               + p(f'{bdg("Breusch-Pagan","red")} Regress ε² on X<br>'
                   f'{bdg("White Test","orange")} Also tests non-linearity<br>'
                   f'{bdg("Scale-Location","blue")} Visual check')
               + p(f'H₀: Var(εᵢ) = σ² (homoscedastic)'),
               "red"),
            ib(f'<span style="color:#ff9f43;-webkit-text-fill-color:#ff9f43;font-weight:600">🔄 Autocorrelation</span><br>'
               + p(f'{bdg("Durbin-Watson","orange")} d ≈ 2 = no autocorr<br>'
                   f'{bdg("Breusch-Godfrey","blue")} Higher-order test<br>'
                   f'{bdg("ACF/PACF Plot","green")} Visual check')
               + p(f'H₀: Cov(εᵢ,εⱼ) = 0 (no autocorrelation)'),
               "orange"),
        )
    )

    # ── 2. Statistical Tests Detail ──────────────────────────────
    render_card("📋 Diagnostic Tests Reference Table",
        table_html(
            ["Test", "H₀", "Statistic", "Rule", "Finance Context"],
            [
                [bdg("Jarque-Bera","gold"),       txt_s("ε ~ Normal"),
                 acc_t("JB = n(S²/6 + (K−3)²/24)"), txt_s("p < 0.05 → Non-normal"),
                 txt_s("Fat tails in return series — common in finance")],
                [bdg("Durbin-Watson","orange"),   txt_s("No autocorrelation"),
                 acc_t("DW = Σ(εₜ−εₜ₋₁)² / Σεₜ²"), txt_s("DW ≈ 2 = no corr; <2 = pos; >2 = neg"),
                 txt_s("Time-series regressions, monthly returns")],
                [bdg("Breusch-Pagan","red"),      txt_s("Homoscedasticity"),
                 acc_t("LM = n·R²_aux"),           txt_s("p < 0.05 → Heteroscedastic"),
                 txt_s("Volatility clustering in equity returns")],
                [bdg("VIF","purple"),              txt_s("No multicollinearity"),
                 acc_t("VIF = 1/(1−Rⱼ²)"),         txt_s("VIF > 10 → Problem"),
                 txt_s("Fama-French factor correlations")],
                [bdg("RESET (Ramsey)","blue"),     txt_s("Correct specification"),
                 acc_t("F = (R²_aug−R²)/(R²_aug...)"),txt_s("p < 0.05 → Misspecified"),
                 txt_s("Non-linear payoffs in options pricing")],
            ]
        )
    )

    # ── 3. Interactive Diagnostic Suite ──────────────────────────
    render_card("🔬 Interactive Diagnostic Suite",
        ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Scenario:</span>'
           + txt_s(' Choose a financial dataset simulation and run the full diagnostic battery. '
                   'Inject heteroscedasticity or autocorrelation to see test responses.'), "gold")
    )

    col1, col2 = st.columns(2)
    scenario = col1.selectbox("Dataset Scenario", [
        "Clean: Well-specified CAPM regression",
        "Heteroscedastic: Volatility clustering (common in equities)",
        "Autocorrelated: Trending residuals (common in interest rates)",
        "Non-normal: Fat tails (returns during crisis)",
        "All violations: Stress test",
    ])
    n_diag = col2.number_input("Sample size (n)", value=100, min_value=30, max_value=300, step=10)
    seed_d = st.number_input("Seed", value=99, min_value=1, step=1, key="diag_seed")

    if st.button("🔬 Run Full Diagnostics", key="diag_run"):
        np.random.seed(int(seed_d)); n = int(n_diag)
        x = np.random.normal(0.006, 0.04, n)
        X = np.column_stack([np.ones(n), x])

        if "Clean" in scenario:
            eps = np.random.normal(0, 0.02, n)
        elif "Heteroscedastic" in scenario:
            eps = np.random.normal(0, 0.015+0.5*np.abs(x), n)
        elif "Autocorrelated" in scenario:
            raw = np.random.normal(0, 0.02, n)
            eps = np.zeros(n); eps[0] = raw[0]
            for t in range(1,n): eps[t] = 0.7*eps[t-1] + raw[t]
        elif "Non-normal" in scenario:
            eps = stats.t.rvs(df=3, scale=0.025, size=n)
        else:
            raw = np.random.normal(0, 0.02, n); eps_ar = np.zeros(n); eps_ar[0] = raw[0]
            for t in range(1,n): eps_ar[t] = 0.5*eps_ar[t-1]+raw[t]
            eps = np.random.normal(0, 0.01+0.4*np.abs(x), n) + eps_ar

        y = 0.002 + 1.1*x + eps
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        yh = X@b; res = y - yh

        # ── Statistical tests ─────────────────────────────────
        jb_stat, jb_p = stats.jarque_bera(res)
        sw_stat, sw_p = stats.shapiro(res[:min(50,n)])
        dw = _durbin_watson(res)
        bp_stat, bp_p = _breusch_pagan(res, X)
        skew = stats.skew(res); kurt = stats.kurtosis(res)

        # Results
        st.html(
            '<div style="margin:12px 0">' +
            table_html(
                ["Test","Statistic","p-value","Decision","Interpretation"],
                [
                    [bdg("Jarque-Bera (Normality)","gold"),
                     hl(f"{jb_stat:.4f}"), txt_s(f"{jb_p:.4f}"),
                     (gt("H₀ Not Rejected ✓") if jb_p>0.05 else rt2("REJECT H₀ ✗")),
                     txt_s("Residuals normal" if jb_p>0.05 else "Non-normal residuals — fat tails/skew")],
                    [bdg("Shapiro-Wilk","purple"),
                     hl(f"{sw_stat:.4f}"), txt_s(f"{sw_p:.4f}"),
                     (gt("H₀ Not Rejected ✓") if sw_p>0.05 else rt2("REJECT H₀ ✗")),
                     txt_s("Normality confirmed" if sw_p>0.05 else "Departure from normality detected")],
                    [bdg("Durbin-Watson","orange"),
                     hl(f"{dw:.4f}"), txt_s("—"),
                     (gt("No Autocorrelation ✓") if 1.5<dw<2.5 else (org("Inconclusive") if abs(dw-2)<0.6 else rt2("Autocorrelation ✗"))),
                     txt_s(f"DW≈2 optimal. Current: {'Pos autocorr' if dw<1.5 else 'Neg autocorr' if dw>2.5 else 'OK'}")],
                    [bdg("Breusch-Pagan","red"),
                     hl(f"{bp_stat:.4f}"), txt_s(f"{bp_p:.4f}"),
                     (gt("Homoscedastic ✓") if bp_p>0.05 else rt2("Heteroscedastic ✗")),
                     txt_s("Constant variance" if bp_p>0.05 else "Variance changes with X — use robust SE")],
                ]
            ) + '</div>'
        )

        metric_row([
            ("Skewness",    f"{skew:.4f}",  None),
            ("Kurtosis",    f"{kurt:.4f}",  None),
            ("DW Statistic",f"{dw:.4f}",   None),
            ("JB p-value",  f"{jb_p:.4f}",  None),
        ])

        fig = _diagnostic_plots(x, y, yh, res, scenario)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Remedies
        violations = []
        if jb_p  < 0.05: violations.append(f'Non-normality detected — consider {hl("robust SE")} or {hl("bootstrap CI")}')
        if bp_p  < 0.05: violations.append(f'Heteroscedasticity — use {hl("WLS")} or {hl("HC robust standard errors")}')
        if not 1.5<dw<2.5: violations.append(f'Autocorrelation (DW={dw:.2f}) — use {hl("HAC/Newey-West SE")} or {hl("AR model")}')

        if violations:
            render_ib(
                f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545;font-weight:600">⚠ Violations Detected — Recommended Remedies:</span><br>'
                + "".join(f'<div style="margin-top:8px;color:#e6f1ff;-webkit-text-fill-color:#e6f1ff">• {v}</div>' for v in violations),
                "red"
            )
        else:
            render_ib(
                f'{gt("✅ All CLRM assumptions satisfied")} — {txt_s("OLS estimates are BLUE. Standard inference is valid.")}',
                "green"
            )


def _durbin_watson(residuals):
    diff = np.diff(residuals)
    return np.sum(diff**2) / np.sum(residuals**2)


def _breusch_pagan(residuals, X):
    n = len(residuals)
    eps2 = residuals**2
    eps2_norm = eps2 / eps2.mean()
    b_aux = np.linalg.lstsq(X, eps2_norm, rcond=None)[0]
    yh_aux = X @ b_aux
    ss_r   = np.sum((yh_aux - eps2_norm.mean())**2)
    ss_t   = np.sum((eps2_norm - eps2_norm.mean())**2)
    r2_aux = ss_r / ss_t if ss_t > 0 else 0
    lm = n * r2_aux
    p  = 1 - stats.chi2.cdf(lm, df=X.shape[1]-1)
    return lm, p


def _diagnostic_plots(x, y, yh, res, scenario):
    fig = plt.figure(figsize=(14, 9), facecolor="#0a1628")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # Residuals vs Fitted
    ax1 = fig.add_subplot(gs[0,0]); ax1.set_facecolor("#112240")
    ax1.scatter(yh*100, res*100, color="#ADD8E6", alpha=0.6, s=35)
    ax1.axhline(0, color="#FFD700", lw=1.5, ls="--")
    ax1.set_xlabel("Fitted (%)", color="#8892b0", fontsize=9)
    ax1.set_ylabel("Residuals (%)", color="#8892b0", fontsize=9)
    ax1.set_title("Residuals vs Fitted", color="#FFD700", fontsize=10)
    _style_ax(ax1)

    # Q-Q Plot
    ax2 = fig.add_subplot(gs[0,1]); ax2.set_facecolor("#112240")
    osm, osr = stats.probplot(res, dist="norm")
    ax2.scatter(osm[0], osm[1]*100, color="#64ffda", alpha=0.7, s=35)
    ax2.plot(osm[0], (osm[0]*osr[0]+osr[1])*100, color="#FFD700", lw=2)
    ax2.set_xlabel("Theoretical Quantiles", color="#8892b0", fontsize=9)
    ax2.set_ylabel("Sample Quantiles", color="#8892b0", fontsize=9)
    ax2.set_title("Q-Q Plot (Normality)", color="#FFD700", fontsize=10)
    _style_ax(ax2)

    # Histogram of residuals
    ax3 = fig.add_subplot(gs[0,2]); ax3.set_facecolor("#112240")
    ax3.hist(res*100, bins=25, color="#004d80", edgecolor="#ADD8E6", alpha=0.8, density=True)
    mu,sg = res.mean()*100, res.std()*100
    xn = np.linspace(mu-4*sg, mu+4*sg, 200)
    ax3.plot(xn, stats.norm.pdf(xn,mu,sg), color="#FFD700", lw=2, label="Normal")
    ax3.legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8)
    ax3.set_xlabel("Residuals (%)", color="#8892b0", fontsize=9)
    ax3.set_title("Residual Distribution", color="#FFD700", fontsize=10)
    _style_ax(ax3)

    # Scale-Location (heteroscedasticity)
    ax4 = fig.add_subplot(gs[1,0]); ax4.set_facecolor("#112240")
    ax4.scatter(yh*100, np.sqrt(np.abs(res*100)), color="#ff9f43", alpha=0.6, s=35)
    ax4.set_xlabel("Fitted (%)", color="#8892b0", fontsize=9)
    ax4.set_ylabel("√|Residuals|", color="#8892b0", fontsize=9)
    ax4.set_title("Scale-Location (Homoscedasticity)", color="#FFD700", fontsize=10)
    _style_ax(ax4)

    # Residuals over time (autocorrelation)
    ax5 = fig.add_subplot(gs[1,1]); ax5.set_facecolor("#112240")
    t = np.arange(len(res))
    ax5.plot(t, res*100, color="#64ffda", lw=1.2, alpha=0.8)
    ax5.axhline(0, color="#FFD700", lw=1, ls="--")
    ax5.fill_between(t, res*100, 0, alpha=0.2, color="#ADD8E6")
    ax5.set_xlabel("Observation", color="#8892b0", fontsize=9)
    ax5.set_ylabel("Residual (%)", color="#8892b0", fontsize=9)
    ax5.set_title("Residuals Over Time (Autocorrelation)", color="#FFD700", fontsize=10)
    _style_ax(ax5)

    # ACF of residuals
    ax6 = fig.add_subplot(gs[1,2]); ax6.set_facecolor("#112240")
    n = len(res); maxlag = min(20, n//4)
    acf_vals = [np.corrcoef(res[:-lag], res[lag:])[0,1] if lag>0 else 1.0 for lag in range(maxlag+1)]
    conf = 1.96/np.sqrt(n)
    ax6.bar(range(maxlag+1), acf_vals, color="#004d80", edgecolor="#ADD8E6", alpha=0.8)
    ax6.axhline(conf,  color="#dc3545", lw=1.5, ls="--", label=f"±1.96/√n")
    ax6.axhline(-conf, color="#dc3545", lw=1.5, ls="--")
    ax6.axhline(0, color="#8892b0", lw=0.5)
    ax6.legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8)
    ax6.set_xlabel("Lag", color="#8892b0", fontsize=9)
    ax6.set_ylabel("ACF", color="#8892b0", fontsize=9)
    ax6.set_title("ACF of Residuals", color="#FFD700", fontsize=10)
    _style_ax(ax6)

    return fig


def _style_ax(ax):
    ax.tick_params(colors="#8892b0", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#1e3a5f")
    ax.grid(color="#1e3a5f", alpha=0.3, lw=0.5)
