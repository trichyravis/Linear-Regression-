"""
tab_slr.py — Simple Linear Regression tab
Financial illustrations: CAPM beta, stock returns, risk-return
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
    lb_t, acc_t, txt_s, p, steps_html, two_col, three_col, four_col,
    table_html, metric_row, section_heading, stat_box, S, FH, FB, FM, TXT, NO_SEL
)


def tab_slr():
    # ── 1. Concept Card ──────────────────────────────────────────
    render_card("📈 Simple Linear Regression — One Predictor, One Outcome",
        p(f'SLR models the {lb_t("<strong>linear relationship</strong>")} between one independent '
          f'variable (X) and one dependent variable (Y). In finance, the most classic application '
          f'is estimating {hl("CAPM Beta")} — how a stock moves relative to the market.')
        + two_col(
            ib(f'<div style="font-family:{FH};color:#FFD700;-webkit-text-fill-color:#FFD700;'
               f'font-size:1.05rem;margin-bottom:8px">📐 The Regression Model</div>'
               + fml("Y = β₀ + β₁X + ε\n\nY  = Dependent variable (e.g. Stock Return)\n"
                     "X  = Independent variable (e.g. Market Return)\nβ₀ = Intercept (Alpha)\n"
                     "β₁ = Slope (Beta) — change in Y per unit X\nε  = Error term (residual)")
               + p(f'{hl("Fitted line:")} Ŷ = β̂₀ + β̂₁X minimises Σ(Yᵢ − Ŷᵢ)²'),
               "gold"),
            ib(f'<div style="font-family:{FH};color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;'
               f'font-size:1.05rem;margin-bottom:8px">💹 CAPM Beta Application</div>'
               + p(f'{lb_t("<strong>Security Characteristic Line (SCL):</strong>")}')
               + fml("Rᵢ − Rf = αᵢ + βᵢ(Rₘ − Rf) + εᵢ\n\nRᵢ = Stock excess return\nRₘ = Market excess return\n"
                     f"α = {hl('Jensen\'s Alpha')} (intercept)\nβ = {hl('Systematic Risk')} (slope)\nε = Idiosyncratic risk")
               + p(f'{bdg("β > 1","red")} Aggressive &nbsp; {bdg("β = 1","gold")} Neutral &nbsp; {bdg("β < 1","green")} Defensive'),
               "blue"),
        )
    )

    # ── 2. OLS Estimation ─────────────────────────────────────────
    render_card("🔧 OLS Estimation — Ordinary Least Squares",
        p(f'OLS finds β̂₀ and β̂₁ by {hl("minimising the sum of squared residuals (SSR)")}. '
          f'The solution is analytical — no iteration required.')
        + two_col(
            fml("β̂₁ = Σ(Xᵢ−X̄)(Yᵢ−Ȳ) / Σ(Xᵢ−X̄)²\n"
                "   = Cov(X,Y) / Var(X)\n"
                "   = r × (Sʏ / Sˣ)\n\nβ̂₀ = Ȳ − β̂₁X̄"),
            fml("SST = Σ(Yᵢ−Ȳ)²    (Total variation)\n"
                "SSR = Σ(Ŷᵢ−Ȳ)²    (Explained)\n"
                "SSE = Σ(Yᵢ−Ŷᵢ)²   (Unexplained)\n\n"
                f"R² = SSR/SST = 1 − SSE/SST\n"
                "SE(β̂₁) = √(MSE / Σ(Xᵢ−X̄)²)")
        )
        + three_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">R² — Goodness of Fit</span><br>'
               + p("Proportion of Y's variance explained by X. Range: 0 to 1.")
               + p(f'{hl("0.8")} → X explains 80% of Y variation'), "gold"),
            ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">t-Test on β̂₁</span><br>'
               + p(f'H₀: β₁ = 0 (no relationship)<br>t = β̂₁ / SE(β̂₁)<br>Reject if |t| > t_crit'), "blue"),
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">F-Test (ANOVA)</span><br>'
               + p(f'H₀: β₁ = 0<br>F = MSR/MSE<br>In SLR: F = t²'), "green"),
        )
    )

    # ── 3. Assumptions ───────────────────────────────────────────
    render_card("⚙ CLRM Assumptions",
        p(f'Classical Linear Regression Model (CLRM) assumptions must hold for OLS to be '
          f'{hl("BLUE")} — Best Linear Unbiased Estimator (Gauss-Markov theorem).')
        + table_html(
            ["#", "Assumption", "What it means", "Violation → Problem"],
            [
                ["1", bdg("Linearity","blue"),        txt_s("E(ε|X) = 0; true model is linear"), txt_s("Biased estimates")],
                ["2", bdg("No Multicollinearity","gold"), txt_s("X variables not perfectly correlated (MLR)"), txt_s("Inflated SE, unstable β̂")],
                ["3", bdg("Homoscedasticity","green"),  txt_s("Var(εᵢ) = σ² (constant variance)"), txt_s("Inefficient OLS, wrong SE")],
                ["4", bdg("No Autocorrelation","orange"),txt_s("Cov(εᵢ,εⱼ) = 0 for i≠j"), txt_s("DW test fails, biased SE")],
                ["5", bdg("Normality","purple"),        txt_s("ε ~ N(0,σ²)"), txt_s("Invalid t/F tests in small samples")],
                ["6", bdg("No Endogeneity","red"),      txt_s("Cov(X,ε) = 0"), txt_s("Biased & inconsistent β̂")],
            ]
        )
    )

    # ── 4. Interactive CAPM Beta Calculator ──────────────────────
    render_card("💹 Interactive CAPM Beta Estimator",
        ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Scenario:</span> '
           + txt_s(' Estimate a stock\'s beta and alpha (Jensen\'s α) by regressing excess stock returns'
                   ' on excess market returns. Uses monthly return data.'), "gold")
    )

    col1, col2, col3 = st.columns([1,1,1])
    stock_name = col1.text_input("Stock Name", value="Infosys (INFY)")
    n_obs      = col2.number_input("No. of Monthly Observations", value=60, min_value=12, max_value=120, step=12)
    true_beta  = col3.number_input("True Beta (for simulation)", value=1.25, min_value=0.1, max_value=3.0, step=0.05)

    col4, col5 = st.columns(2)
    risk_free = col4.number_input("Risk-Free Rate (% annual)", value=6.5, step=0.1)
    seed_val  = col5.number_input("Random Seed", value=42, min_value=1, step=1)

    if st.button("📊 Run CAPM Regression", key="capm_run"):
        np.random.seed(int(seed_val))
        rf_monthly  = risk_free / 12 / 100
        mkt_excess  = np.random.normal(0.006, 0.045, n_obs)
        true_alpha  = 0.002
        epsilon     = np.random.normal(0, 0.03, n_obs)
        stock_excess = true_alpha + true_beta * mkt_excess + epsilon

        x = mkt_excess; y = stock_excess
        n = len(x)
        x_bar, y_bar = x.mean(), y.mean()
        beta1_hat = np.sum((x-x_bar)*(y-y_bar)) / np.sum((x-x_bar)**2)
        beta0_hat = y_bar - beta1_hat * x_bar
        y_hat     = beta0_hat + beta1_hat * x
        residuals = y - y_hat

        sse   = np.sum(residuals**2)
        sst   = np.sum((y-y_bar)**2)
        ssr   = sst - sse
        r2    = ssr / sst
        mse   = sse / (n-2)
        se_b1 = np.sqrt(mse / np.sum((x-x_bar)**2))
        se_b0 = np.sqrt(mse * (1/n + x_bar**2/np.sum((x-x_bar)**2)))
        t_b1  = beta1_hat / se_b1
        t_b0  = beta0_hat / se_b0
        p_b1  = 2*(1-stats.t.cdf(abs(t_b1), df=n-2))
        p_b0  = 2*(1-stats.t.cdf(abs(t_b0), df=n-2))
        f_stat= (ssr/1) / mse
        adj_r2= 1-(1-r2)*(n-1)/(n-2)
        corr  = np.corrcoef(x,y)[0,1]

        # Metrics
        metric_row([
            ("Beta (β̂₁)",              f"{beta1_hat:.4f}", None),
            ("Alpha β̂₀ (monthly)",     f"{beta0_hat*100:.4f}%", None),
            ("R²",                      f"{r2:.4f}", None),
            ("Adjusted R²",             f"{adj_r2:.4f}", None),
        ])
        metric_row([
            ("t-stat (Beta)",           f"{t_b1:.4f}", None),
            ("p-value (Beta)",          f"{p_b1:.4f}", None),
            ("F-statistic",             f"{f_stat:.4f}", None),
            ("Correlation (r)",         f"{corr:.4f}", None),
        ])

        # Regression table
        t_crit = stats.t.ppf(0.975, df=n-2)
        ci_b1_lo = beta1_hat - t_crit*se_b1; ci_b1_hi = beta1_hat + t_crit*se_b1
        ci_b0_lo = beta0_hat - t_crit*se_b0; ci_b0_hi = beta0_hat + t_crit*se_b0

        st.html(table_html(
            ["Parameter","Estimate","Std Error","t-stat","p-value","95% CI","Significance"],
            [
                [txt_s("β̂₀ (Alpha)"), hl(f"{beta0_hat*100:.4f}%"),
                 txt_s(f"{se_b0*100:.4f}%"), txt_s(f"{t_b0:.4f}"),
                 (gt("sig ✓") if p_b0<0.05 else txt_s(f"{p_b0:.4f}")),
                 txt_s(f"[{ci_b0_lo*100:.4f}%, {ci_b0_hi*100:.4f}%]"),
                 bdg("Sig","green") if p_b0<0.05 else bdg("Not Sig","red")],
                [txt_s("β̂₁ (Beta)"), hl(f"{beta1_hat:.4f}"),
                 txt_s(f"{se_b1:.4f}"), txt_s(f"{t_b1:.4f}"),
                 (gt(f"{p_b1:.4f}") if p_b1<0.05 else rt2(f"{p_b1:.4f}")),
                 txt_s(f"[{ci_b1_lo:.4f}, {ci_b1_hi:.4f}]"),
                 bdg("Sig","green") if p_b1<0.05 else bdg("Not Sig","red")],
            ]
        ))

        # Plots
        fig = _slr_plots(x, y, y_hat, residuals, stock_name, beta1_hat, beta0_hat)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Interpretation
        beta_type = "Aggressive (β>1)" if beta1_hat>1.05 else ("Defensive (β<1)" if beta1_hat<0.95 else "Market-Neutral (β≈1)")
        alpha_ann = beta0_hat * 12 * 100
        render_ib(
            f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">📊 Interpretation for {stock_name}:</span><br>'
            + steps_html([
                ("Beta Interpretation",
                 f'{hl(f"β = {beta1_hat:.4f}")} → {txt_s(beta_type)}. '
                 + txt_s(f'For every 1% market move, stock moves {beta1_hat*100:.2f} bps. '
                         f'True beta = {true_beta:.2f}; estimation error = {abs(beta1_hat-true_beta):.4f}.')),
                ("Alpha (Jensen's α)",
                 txt_s(f'Monthly α = {beta0_hat*100:.4f}% → Annualised = {alpha_ann:.4f}%. ')
                 + (gt("Positive alpha — stock generates excess risk-adjusted return! ✓")
                    if beta0_hat > 0 else rt2("Negative alpha — underperforms risk-adjusted benchmark."))),
                ("R² Interpretation",
                 txt_s(f'R² = {r2:.4f} → Market explains ') + hl(f"{r2*100:.1f}%")
                 + txt_s(f' of this stock\'s return variation. Residual {(1-r2)*100:.1f}% is firm-specific risk.')),
                ("Hypothesis Test",
                 txt_s(f't = {t_b1:.4f}, p = {p_b1:.4f}. ')
                 + (gt("REJECT H₀: β₁ = 0 → Significant market relationship. ✓")
                    if p_b1 < 0.05 else rt2("FAIL TO REJECT H₀: β₁ not significantly different from 0."))),
            ]),
            "gold"
        )


def _slr_plots(x, y, y_hat, residuals, stock_name, beta1, beta0):
    fig = plt.figure(figsize=(14, 10), facecolor="#0a1628")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Scatter + Regression Line ─────────────────────────────
    ax1 = fig.add_subplot(gs[0,:2])
    ax1.set_facecolor("#112240")
    ax1.scatter(x*100, y*100, color="#64ffda", alpha=0.6, s=40, zorder=3, label="Monthly Returns")
    x_line = np.linspace(x.min(), x.max(), 200)
    ax1.plot(x_line*100, (beta0+beta1*x_line)*100, color="#FFD700", lw=2.5, zorder=4, label=f"SCL: α={beta0*100:.3f}%, β={beta1:.4f}")
    ax1.axhline(0, color="#8892b0", lw=0.5, ls="--"); ax1.axvline(0, color="#8892b0", lw=0.5, ls="--")
    ax1.set_xlabel("Market Excess Return (%)", color="#8892b0", fontsize=9)
    ax1.set_ylabel(f"{stock_name} Excess Return (%)", color="#8892b0", fontsize=9)
    ax1.set_title("Security Characteristic Line (SCL)", color="#FFD700", fontsize=11)
    ax1.legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")
    _style_ax(ax1)

    # ── Residuals vs Fitted ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0,2])
    ax2.set_facecolor("#112240")
    ax2.scatter(y_hat*100, residuals*100, color="#ADD8E6", alpha=0.6, s=35)
    ax2.axhline(0, color="#FFD700", lw=1.5, ls="--")
    ax2.set_xlabel("Fitted Values (%)", color="#8892b0", fontsize=9)
    ax2.set_ylabel("Residuals (%)", color="#8892b0", fontsize=9)
    ax2.set_title("Residuals vs Fitted", color="#FFD700", fontsize=11)
    _style_ax(ax2)

    # ── Residual Histogram ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1,0])
    ax3.set_facecolor("#112240")
    ax3.hist(residuals*100, bins=20, color="#004d80", edgecolor="#ADD8E6", alpha=0.8)
    mu, sigma = residuals.mean()*100, residuals.std()*100
    xn = np.linspace(mu-4*sigma, mu+4*sigma, 200)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(xn, stats.norm.pdf(xn, mu, sigma), color="#FFD700", lw=2)
    ax3_twin.set_yticks([]); ax3_twin.set_facecolor("#112240")
    ax3.set_xlabel("Residuals (%)", color="#8892b0", fontsize=9)
    ax3.set_ylabel("Frequency", color="#8892b0", fontsize=9)
    ax3.set_title("Residual Distribution", color="#FFD700", fontsize=11)
    _style_ax(ax3)

    # ── Q-Q Plot ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1,1])
    ax4.set_facecolor("#112240")
    osm, osr = stats.probplot(residuals, dist="norm")
    ax4.scatter(osm[0], osm[1]*100, color="#64ffda", alpha=0.7, s=35)
    ax4.plot(osm[0], (osm[0]*osr[0]+osr[1])*100, color="#FFD700", lw=2)
    ax4.set_xlabel("Theoretical Quantiles", color="#8892b0", fontsize=9)
    ax4.set_ylabel("Sample Quantiles (%)", color="#8892b0", fontsize=9)
    ax4.set_title("Q-Q Plot (Normality Check)", color="#FFD700", fontsize=11)
    _style_ax(ax4)

    # ── √|Residuals| vs Fitted (Scale-Location) ───────────────
    ax5 = fig.add_subplot(gs[1,2])
    ax5.set_facecolor("#112240")
    ax5.scatter(y_hat*100, np.sqrt(np.abs(residuals*100)), color="#ff9f43", alpha=0.6, s=35)
    ax5.set_xlabel("Fitted Values (%)", color="#8892b0", fontsize=9)
    ax5.set_ylabel("√|Residuals|", color="#8892b0", fontsize=9)
    ax5.set_title("Scale-Location (Homoscedasticity)", color="#FFD700", fontsize=11)
    _style_ax(ax5)

    return fig


def _style_ax(ax):
    ax.tick_params(colors="#8892b0", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#1e3a5f")
    ax.grid(color="#1e3a5f", alpha=0.3, lw=0.5)
