"""
tab_finance.py — Applied Finance Regression Cases
Bond yield modelling, Credit risk scoring, Equity valuation
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


def tab_finance():
    render_card("🏦 Applied Finance Regression Cases",
        p(f'Real-world applications of SLR and MLR across {hl("fixed income")}, '
          f'{hl("credit risk")}, and {hl("equity valuation")} domains.')
        + three_col(
            ib(f'<div style="font-family:{FH};color:#FFD700;-webkit-text-fill-color:#FFD700;font-size:1rem;margin-bottom:6px">📐 Bond Yield Model (SLR)</div>'
               + fml("Yield = β₀ + β₁(Duration) + ε\n\nDuration = key X variable\nHigher duration → higher yield premium\nFor convexity → add Duration²")
               + p(f'{bdg("Fixed Income","blue")} {bdg("Duration Risk","red")}'), "blue"),
            ib(f'<div style="font-family:{FH};color:#28a745;-webkit-text-fill-color:#28a745;font-size:1rem;margin-bottom:6px">💳 Credit Scoring (MLR)</div>'
               + fml("PD = β₀ + β₁(D/E) + β₂(ICR)\n    + β₃(CurrentRatio) + ε\n\nPD = Probability of Default\nD/E = Leverage\nICR = Interest Coverage")
               + p(f'{bdg("Credit Risk","red")} {bdg("Altman Z-Score","gold")}'), "green"),
            ib(f'<div style="font-family:{FH};color:#ff9f43;-webkit-text-fill-color:#ff9f43;font-size:1rem;margin-bottom:6px">📊 P/E Ratio Model (MLR)</div>'
               + fml("P/E = β₀ + β₁(ROE) + β₂(g)\n    + β₃(Beta) + ε\n\nROE = Return on Equity\ng   = Expected growth rate\nBeta = Systematic risk")
               + p(f'{bdg("Equity Valuation","gold")} {bdg("Gordon Growth","green")}'), "orange"),
        )
    )

    case = st.radio("Choose Case Study",
                    ["📐 Bond Yield vs Duration (SLR)",
                     "💳 Credit Risk Scoring (MLR)",
                     "📊 Equity P/E Valuation (MLR)"],
                    horizontal=True, key="fin_case")

    if "Bond" in case:
        _bond_case()
    elif "Credit" in case:
        _credit_case()
    else:
        _pe_case()


def _bond_case():
    render_card("📐 Bond Yield vs Modified Duration — SLR",
        ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Context:</span>'
           + txt_s(' In a normal upward-sloping yield curve, bonds with higher modified duration '
                   'demand higher yields as compensation for interest rate risk. '
                   'SLR quantifies this yield-duration relationship.'), "gold")
    )

    col1, col2, col3 = st.columns(3)
    n_bonds  = col1.number_input("Number of bonds", value=40, min_value=10, max_value=100, step=5)
    base_yld = col2.number_input("Base Yield (%)  ", value=6.5, step=0.5)
    seed_b   = col3.number_input("Seed", value=21, min_value=1, step=1, key="bond_seed")

    if st.button("📐 Run Bond Regression", key="bond_run"):
        np.random.seed(int(seed_b)); n = int(n_bonds)
        duration = np.random.uniform(0.5, 12, n)
        true_b1  = 0.25
        epsilon  = np.random.normal(0, 0.30, n)
        yield_pct = base_yld + true_b1 * duration + epsilon

        x, y = duration, yield_pct
        xb, yb = x.mean(), y.mean()
        b1 = np.sum((x-xb)*(y-yb))/np.sum((x-xb)**2)
        b0 = yb - b1*xb
        yh = b0 + b1*x; res = y - yh
        n_ = len(x); sse = np.sum(res**2); sst = np.sum((y-yb)**2)
        r2 = 1-sse/sst; mse = sse/(n_-2); se_b1 = np.sqrt(mse/np.sum((x-xb)**2))
        t1 = b1/se_b1; p1 = 2*(1-stats.t.cdf(abs(t1), df=n_-2))

        metric_row([
            ("Intercept β̂₀ (base yield)", f"{b0:.4f}%", None),
            ("Slope β̂₁ (yield/year dur.)",  f"{b1:.4f}%", None),
            ("R²",                            f"{r2:.4f}", None),
            ("t-stat (β̂₁)",                 f"{t1:.4f}", None),
        ])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="#0a1628")
        for ax in axes: ax.set_facecolor("#112240")
        axes[0].scatter(duration, yield_pct, color="#64ffda", alpha=0.7, s=50, label="Bonds")
        xl = np.linspace(duration.min(), duration.max(), 200)
        axes[0].plot(xl, b0+b1*xl, color="#FFD700", lw=2.5, label=f"Yield = {b0:.2f} + {b1:.4f}×Duration")
        axes[0].set_xlabel("Modified Duration (years)", color="#8892b0", fontsize=9)
        axes[0].set_ylabel("Yield to Maturity (%)", color="#8892b0", fontsize=9)
        axes[0].set_title("Bond Yield vs Modified Duration", color="#FFD700", fontsize=11)
        axes[0].legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")
        _sax(axes[0])

        axes[1].scatter(yh, res, color="#ADD8E6", alpha=0.7, s=40)
        axes[1].axhline(0, color="#FFD700", lw=1.5, ls="--")
        axes[1].set_xlabel("Fitted Yield (%)", color="#8892b0", fontsize=9)
        axes[1].set_ylabel("Residuals (%)", color="#8892b0", fontsize=9)
        axes[1].set_title("Residuals vs Fitted", color="#FFD700", fontsize=11)
        _sax(axes[1])
        plt.tight_layout(pad=1.5); st.pyplot(fig, use_container_width=True); plt.close(fig)

        render_ib(
            f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Interpretation:</span><br>'
            + steps_html([
                ("Yield-Duration Slope",
                 txt_s(f'β̂₁ = {b1:.4f}% per year of duration. Each additional year of duration '
                       f'demands {hl(f"{b1*100:.2f} bps")} of additional yield compensation.')),
                ("Base Rate",
                 txt_s(f'β̂₀ = {b0:.4f}% — the theoretical yield for a zero-duration instrument '
                       f'(approximately the risk-free rate for this curve).')),
                ("Explanatory Power",
                 txt_s(f'R² = {r2:.4f} → Duration explains {hl(f"{r2*100:.1f}%")} of cross-sectional yield variation. '
                       f'Residual variation reflects credit spreads, liquidity, and coupon effects.')),
            ]), "gold"
        )


def _credit_case():
    render_card("💳 Credit Risk Regression — PD Modelling",
        ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Context:</span>'
           + txt_s(' Model probability of default (PD) using financial ratios from a corporate loan portfolio. '
                   'Based on Altman Z-score framework. Uses OLS for illustration '
                   '(logistic regression is more appropriate for binary outcomes).'), "gold")
    )

    col1, col2 = st.columns(2)
    n_firms = col1.number_input("Number of firms", value=50, min_value=20, max_value=150, step=10)
    seed_c  = col2.number_input("Seed", value=33, min_value=1, step=1, key="cr_seed")

    if st.button("💳 Run Credit Risk Regression", key="cr_run"):
        np.random.seed(int(seed_c)); n = int(n_firms)
        de_ratio  = np.random.uniform(0.2, 4.0, n)
        icr       = np.random.uniform(0.5, 8.0, n)
        curr_r    = np.random.uniform(0.5, 3.5, n)
        roa       = np.random.uniform(-0.05, 0.15, n)
        eps       = np.random.normal(0, 0.04, n)
        pd_score  = np.clip(0.05 + 0.08*de_ratio - 0.04*icr - 0.06*curr_r - 0.3*roa + eps, 0, 0.9)

        X = np.column_stack([np.ones(n), de_ratio, icr, curr_r, roa])
        b = np.linalg.lstsq(X, pd_score, rcond=None)[0]
        yh = X@b; res = pd_score - yh
        sse = np.sum(res**2); sst = np.sum((pd_score-pd_score.mean())**2)
        r2 = 1-sse/sst; k = 4
        adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)
        mse = sse/(n-k-1)
        XtX_inv = np.linalg.inv(X.T@X)
        se_all = np.sqrt(mse*np.diag(XtX_inv))
        t_vals = b/se_all
        p_vals = [2*(1-stats.t.cdf(abs(t), df=n-k-1)) for t in t_vals]

        metric_row([
            ("R²",        f"{r2:.4f}",    None),
            ("Adj R²",    f"{adj_r2:.4f}", None),
            ("MSE",       f"{mse:.6f}",   None),
            ("N firms",   f"{n}",          None),
        ])

        names = ["Intercept","D/E Ratio","Interest Coverage","Current Ratio","ROA"]
        rows = []
        for nm,bv,se,tv,pv in zip(names,b,se_all,t_vals,p_vals):
            sig = pv < 0.05
            expected_sign = {"D/E Ratio":"+","Interest Coverage":"−","Current Ratio":"−","ROA":"−"}.get(nm,"")
            rows.append([
                txt_s(nm),
                hl(f"{bv:.4f}") if sig else txt_s(f"{bv:.4f}"),
                txt_s(f"{se:.4f}"), txt_s(f"{tv:.4f}"),
                gt(f"{pv:.4f}") if sig else rt2(f"{pv:.4f}"),
                txt_s(expected_sign),
                bdg("***","green") if pv<0.001 else (bdg("**","gold") if pv<0.01 else (bdg("*","orange") if pv<0.05 else bdg("ns","red")))
            ])
        st.html(table_html(["Variable","Coefficient","SE","t-stat","p-value","Expected Sign","Sig"], rows))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0a1628")
        for ax in axes: ax.set_facecolor("#112240")

        axes[0].scatter(yh, pd_score, color="#64ffda", alpha=0.7, s=40)
        mn,mx = min(yh.min(),pd_score.min()), max(yh.max(),pd_score.max())
        axes[0].plot([mn,mx],[mn,mx], color="#FFD700", lw=2, ls="--")
        axes[0].set_xlabel("Predicted PD", color="#8892b0", fontsize=9)
        axes[0].set_ylabel("Actual PD", color="#8892b0", fontsize=9)
        axes[0].set_title("Predicted vs Actual PD", color="#FFD700", fontsize=11)
        _sax(axes[0])

        colors_c = ["#FFD700","#28a745","#ff9f43","#a29bfe"]
        axes[1].barh(range(4), b[1:], color=colors_c, alpha=0.85, height=0.5)
        axes[1].set_yticks(range(4)); axes[1].set_yticklabels(names[1:], color="#e6f1ff", fontsize=8)
        axes[1].axvline(0, color="#8892b0", lw=1, ls="--")
        axes[1].set_xlabel("Coefficient", color="#8892b0", fontsize=9)
        axes[1].set_title("PD Drivers", color="#FFD700", fontsize=11)
        _sax(axes[1])

        axes[2].scatter(yh, res, color="#ADD8E6", alpha=0.6, s=35)
        axes[2].axhline(0, color="#FFD700", lw=1.5, ls="--")
        axes[2].set_xlabel("Fitted PD", color="#8892b0", fontsize=9); axes[2].set_ylabel("Residuals", color="#8892b0", fontsize=9)
        axes[2].set_title("Residuals vs Fitted", color="#FFD700", fontsize=11)
        _sax(axes[2])
        plt.tight_layout(pad=1.5); st.pyplot(fig, use_container_width=True); plt.close(fig)


def _pe_case():
    render_card("📊 Equity P/E Valuation Model",
        ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Context:</span>'
           + txt_s(' Model the cross-sectional variation in P/E ratios across stocks using fundamental '
                   'drivers from the Gordon Growth Model: ROE, expected growth, and systematic risk (beta). '
                   'Helps identify over/under-valued stocks.'), "gold")
    )
    col1, col2 = st.columns(2)
    n_stocks = col1.number_input("Number of stocks", value=60, min_value=20, max_value=150, step=10)
    seed_p   = col2.number_input("Seed", value=55, min_value=1, step=1, key="pe_seed")

    if st.button("📊 Run P/E Regression", key="pe_run"):
        np.random.seed(int(seed_p)); n = int(n_stocks)
        roe    = np.random.uniform(0.05, 0.30, n)
        g      = np.random.uniform(0.02, 0.18, n)
        beta   = np.random.uniform(0.4, 2.2, n)
        divpay = np.random.uniform(0.2, 0.7, n)
        eps_pe = np.random.normal(0, 3, n)
        pe     = np.clip(5 + 80*roe + 60*g - 8*beta + 15*divpay + eps_pe, 2, 60)

        X = np.column_stack([np.ones(n), roe, g, beta, divpay])
        b = np.linalg.lstsq(X, pe, rcond=None)[0]
        yh = X@b; res = pe - yh
        sse = np.sum(res**2); sst = np.sum((pe-pe.mean())**2)
        r2 = 1-sse/sst; k=4; adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)
        mse = sse/(n-k-1)
        se_all = np.sqrt(mse*np.diag(np.linalg.inv(X.T@X)))
        t_vals = b/se_all; p_vals = [2*(1-stats.t.cdf(abs(t),df=n-k-1)) for t in t_vals]

        metric_row([
            ("R²",       f"{r2:.4f}",     None),
            ("Adj R²",   f"{adj_r2:.4f}",  None),
            ("Avg P/E",  f"{pe.mean():.2f}x", None),
            ("RMSE",     f"{np.sqrt(mse):.2f}x", None),
        ])

        names = ["Intercept","ROE","Growth (g)","Beta (β)","Dividend Payout"]
        rows = []
        for nm,bv,se,tv,pv in zip(names,b,se_all,t_vals,p_vals):
            sig = pv<0.05
            sign = {"ROE":"+","Growth (g)":"+","Beta (β)":"−","Dividend Payout":"+"}.get(nm,"")
            rows.append([txt_s(nm), hl(f"{bv:.4f}") if sig else txt_s(f"{bv:.4f}"),
                         txt_s(f"{se:.4f}"), txt_s(f"{tv:.4f}"),
                         gt(f"{pv:.4f}") if sig else rt2(f"{pv:.4f}"),
                         txt_s(sign),
                         bdg("***","green") if pv<0.001 else (bdg("**","gold") if pv<0.01 else (bdg("*","orange") if pv<0.05 else bdg("ns","red")))])
        st.html(table_html(["Variable","Coefficient","SE","t-stat","p-value","Exp. Sign","Sig"], rows))

        fig, axes = plt.subplots(1,3, figsize=(14,4.5), facecolor="#0a1628")
        for ax in axes: ax.set_facecolor("#112240")

        axes[0].scatter(yh, pe, color="#FFD700", alpha=0.7, s=50)
        mn,mx = min(yh.min(),pe.min()),max(yh.max(),pe.max())
        axes[0].plot([mn,mx],[mn,mx], color="#64ffda", lw=2, ls="--", label="Fair Value Line")
        axes[0].set_xlabel("Model P/E", color="#8892b0", fontsize=9); axes[0].set_ylabel("Actual P/E", color="#8892b0", fontsize=9)
        axes[0].set_title("Model vs Actual P/E", color="#FFD700", fontsize=11)
        axes[0].legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")
        _sax(axes[0])

        # Mispricing (residuals as alpha signal)
        top5 = np.argsort(res)[-5:]  # most undervalued
        bot5 = np.argsort(res)[:5]   # most overvalued
        color_arr = np.where(res>0,"#28a745","#dc3545")
        axes[1].bar(range(n), sorted(res), color=sorted(color_arr, key=lambda _: 0), alpha=0.7, width=1)
        axes[1].axhline(0, color="#FFD700", lw=1.5, ls="--")
        axes[1].set_xlabel("Stocks (sorted)", color="#8892b0", fontsize=9); axes[1].set_ylabel("P/E Residual (Mispricing)", color="#8892b0", fontsize=9)
        axes[1].set_title("Mispricing Signal (Green=Undervalued)", color="#FFD700", fontsize=11)
        _sax(axes[1])

        axes[2].scatter(roe*100, pe, color="#a29bfe", alpha=0.7, s=40, label="Stocks")
        xr = np.linspace(roe.min(),roe.max(),100)
        slope = np.polyfit(roe,pe,1)
        axes[2].plot(xr*100, np.polyval(slope,xr), color="#FFD700", lw=2, label=f"Trend")
        axes[2].set_xlabel("ROE (%)", color="#8892b0", fontsize=9); axes[2].set_ylabel("P/E Ratio", color="#8892b0", fontsize=9)
        axes[2].set_title("P/E vs ROE", color="#FFD700", fontsize=11)
        axes[2].legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")
        _sax(axes[2])
        plt.tight_layout(pad=1.5); st.pyplot(fig, use_container_width=True); plt.close(fig)


def _sax(ax):
    ax.tick_params(colors="#8892b0", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#1e3a5f")
    ax.grid(color="#1e3a5f", alpha=0.3, lw=0.5)
