"""
tab_mlr.py — Multiple Linear Regression tab
Financial illustrations: Fama-French factors, credit risk, bond pricing
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


def tab_mlr():
    # ── 1. Concept ───────────────────────────────────────────────
    render_card("📊 Multiple Linear Regression — k Predictors",
        p(f'MLR extends SLR to {lb_t("<strong>multiple independent variables</strong>")} simultaneously, '
          f'controlling for each variable\'s effect while holding others constant. '
          f'In finance: {hl("Fama-French 3-factor model")}, credit scoring, bond yield modelling.')
        + two_col(
            ib(f'<div style="font-family:{FH};color:#FFD700;-webkit-text-fill-color:#FFD700;font-size:1.05rem;margin-bottom:8px">📐 General MLR Model</div>'
               + fml("Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε\n\n"
                     "In matrix form:\n  Y = Xβ + ε\n\n"
                     "OLS Solution:\n  β̂ = (XᵀX)⁻¹ Xᵀ Y\n\n"
                     "Var(β̂) = σ²(XᵀX)⁻¹"),
               "gold"),
            ib(f'<div style="font-family:{FH};color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-size:1.05rem;margin-bottom:8px">🏦 Fama-French 3-Factor Model</div>'
               + fml("Rᵢ−Rf = αᵢ + β₁(Rₘ−Rf) + β₂SMB + β₃HML + εᵢ\n\n"
                     "Rₘ−Rf  = Market risk premium\n"
                     "SMB    = Small Minus Big (size factor)\n"
                     "HML    = High Minus Low (value factor)\n"
                     "α      = Unexplained excess return\n\n"
                     "Extension: Carhart + Momentum factor")
               + p(f'{bdg("Market","blue")} {bdg("Size (SMB)","gold")} {bdg("Value (HML)","green")} {bdg("Momentum","orange")}'),
               "blue"),
        )
    )

    # ── 2. R², Adj R², F-test ────────────────────────────────────
    render_card("📏 Model Fit — R², Adjusted R², F-Test & Information Criteria",
        three_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">R² vs Adjusted R²</span><br>'
               + fml("R²    = 1 − SSE/SST\n\nAdj R² = 1 − (1−R²)(n−1)/(n−k−1)\n\nAdj R² penalises irrelevant variables")
               + p(f'{rt2("⚠")} R² never decreases when adding variables. Always use {hl("Adj R²")} in MLR.'),
               "gold"),
            ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">F-Statistic (Overall)</span><br>'
               + fml("H₀: β₁=β₂=...=βₖ=0\n\nF = (R²/k) / ((1−R²)/(n−k−1))\n  = MSR / MSE\n\np-value < α → Model is significant")
               + p(f'df₁ = k (numerator), df₂ = n−k−1 (denominator)'),
               "blue"),
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">AIC & BIC</span><br>'
               + fml("AIC = 2k − 2ln(L)\nBIC = k·ln(n) − 2ln(L)\n\nL = maximised log-likelihood\nk = no. of parameters")
               + p(f'Lower AIC/BIC = better. {hl("BIC penalises k more heavily")} — preferred for model selection.'),
               "green"),
        )
    )

    # ── 3. Multicollinearity ─────────────────────────────────────
    render_card("⚠ Multicollinearity — Detection & Remedies",
        two_col(
            ib(f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545;font-weight:600">What is Multicollinearity?</span><br>'
               + p("When two or more X variables are highly correlated, OLS estimates become "
                   "unstable and standard errors inflate — even though R² remains high.")
               + fml("VIF = 1 / (1−Rⱼ²)\n\nRⱼ² = R² from regressing Xⱼ on all other Xs\n\n"
                     "VIF > 5  → Moderate concern\nVIF > 10 → Serious multicollinearity")
               + p(f'{rt2("Finance example:")} Including both Nifty 50 and Nifty 500 as regressors — nearly identical series.'),
               "red"),
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Detection & Remedies</span><br>'
               + table_html(
                   ["Method","When to use"],
                   [[bdg("Correlation matrix","blue"), txt_s("Pairwise check — r > 0.8 signals risk")],
                    [bdg("VIF","red"),          txt_s("VIF > 10 → drop or combine variable")],
                    [bdg("Ridge regression","orange"), txt_s("Adds λΣβᵢ² penalty to shrink coefficients")],
                    [bdg("PCA","purple"),       txt_s("Combine correlated Xs into orthogonal components")],
                    [bdg("More data","green"),  txt_s("Increases sample variance, reduces SE")]],
               ),
               "gold"),
        )
    )

    # ── 4. Interactive Fama-French Regression ────────────────────
    render_card("🏦 Interactive Fama-French Multi-Factor Regression",
        ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Scenario:</span>'
           + txt_s(' Regress a fund\'s excess returns on market, size (SMB), and value (HML) factors. '
                   'Identify which factors drive performance and estimate risk-adjusted alpha.'), "gold")
    )

    col1, col2, col3 = st.columns(3)
    fund_name  = col1.text_input("Fund/Stock Name", value="ABC Equity Fund")
    n_obs      = col2.number_input("Monthly Observations", value=60, min_value=24, max_value=120, step=12)
    seed_v     = col3.number_input("Random Seed", value=7, min_value=1, step=1)

    col4, col5, col6, col7 = st.columns(4)
    true_b_mkt = col4.number_input("True β_MKT", value=0.95, min_value=0.0, max_value=3.0, step=0.05)
    true_b_smb = col5.number_input("True β_SMB", value=0.40, min_value=-1.0, max_value=2.0, step=0.05)
    true_b_hml = col6.number_input("True β_HML", value=0.30, min_value=-1.0, max_value=2.0, step=0.05)
    include_mom= col7.checkbox("Include Momentum (WML)", value=False)

    if st.button("📊 Run Fama-French Regression", key="ff_run"):
        np.random.seed(int(seed_v))
        mkt  = np.random.normal(0.006, 0.045, n_obs)
        smb  = np.random.normal(0.002, 0.025, n_obs)
        hml  = np.random.normal(0.002, 0.022, n_obs)
        wml  = np.random.normal(0.004, 0.032, n_obs)
        eps  = np.random.normal(0, 0.018, n_obs)
        true_alpha = 0.0018

        fund = true_alpha + true_b_mkt*mkt + true_b_smb*smb + true_b_hml*hml + eps
        if include_mom:
            true_b_wml = 0.15
            fund += true_b_wml * wml

        factors = [mkt, smb, hml] + ([wml] if include_mom else [])
        factor_names = ["Market (Rₘ−Rf)", "SMB (Size)", "HML (Value)"] + (["WML (Momentum)"] if include_mom else [])
        k = len(factors)

        X = np.column_stack([np.ones(n_obs)] + factors)
        Y = fund
        beta_hat  = np.linalg.lstsq(X, Y, rcond=None)[0]
        y_hat     = X @ beta_hat
        residuals = Y - y_hat
        n = n_obs

        sse   = np.sum(residuals**2)
        sst   = np.sum((Y-Y.mean())**2)
        ssr   = sst - sse
        r2    = ssr/sst
        adj_r2= 1-(1-r2)*(n-1)/(n-k-1)
        mse   = sse/(n-k-1)
        f_stat= (ssr/k)/mse
        p_f   = 1-stats.f.cdf(f_stat, k, n-k-1)
        aic   = n*np.log(sse/n) + 2*(k+1)
        bic   = n*np.log(sse/n) + (k+1)*np.log(n)

        XtX_inv = np.linalg.inv(X.T @ X)
        se_all  = np.sqrt(mse * np.diag(XtX_inv))
        t_vals  = beta_hat / se_all
        p_vals  = [2*(1-stats.t.cdf(abs(t), df=n-k-1)) for t in t_vals]
        t_crit  = stats.t.ppf(0.975, df=n-k-1)
        vifs = _compute_vif(factors)

        # Metrics row
        metric_row([
            ("R²",               f"{r2:.4f}",    None),
            ("Adjusted R²",      f"{adj_r2:.4f}", None),
            ("F-Statistic",      f"{f_stat:.2f}", None),
            ("p(F)",             f"{p_f:.4f}",    None),
        ])
        metric_row([
            ("AIC",              f"{aic:.2f}",    None),
            ("BIC",              f"{bic:.2f}",    None),
            ("MSE",              f"{mse*10000:.4f} bps²", None),
            ("RMSE",             f"{np.sqrt(mse)*100:.4f}%", None),
        ])

        # Coefficient table
        all_names = ["Intercept (α)"] + factor_names
        coeff_rows = []
        for i,(nm,b,se,tv,pv) in enumerate(zip(all_names,beta_hat,se_all,t_vals,p_vals)):
            ci_lo = b-t_crit*se; ci_hi = b+t_crit*se
            sig = pv < 0.05
            b_str = f"{b*100:.4f}%" if i==0 else hl(f"{b:.4f}") if sig else txt_s(f"{b:.4f}")
            coeff_rows.append([
                txt_s(nm), b_str, txt_s(f"{se:.4f}"),
                txt_s(f"{tv:.4f}"),
                gt(f"{pv:.4f}") if sig else rt2(f"{pv:.4f}"),
                txt_s(f"[{ci_lo:.4f}, {ci_hi:.4f}]"),
                bdg("***","green") if pv<0.001 else (bdg("**","gold") if pv<0.01 else (bdg("*","orange") if pv<0.05 else bdg("ns","red")))
            ])
        st.html(table_html(["Parameter","Estimate","Std Error","t-stat","p-value","95% CI","Sig"], coeff_rows))

        # VIF Table
        vif_rows = [[txt_s(nm), (gt(f"{v:.2f}") if v<5 else (org(f"{v:.2f}") if v<10 else rt2(f"{v:.2f}"))),
                     bdg("OK","green") if v<5 else (bdg("Moderate","orange") if v<10 else bdg("High","red"))]
                    for nm,v in zip(factor_names, vifs)]
        st.html('<div style="margin-top:12px">' +
                table_html(["Factor","VIF","Status"], vif_rows) + '</div>')

        # Plots
        fig = _mlr_plots(Y*100, y_hat*100, residuals*100, beta_hat[1:], se_all[1:], factor_names, fund_name)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Interpretation
        alpha_ann = beta_hat[0]*12*100
        dominant = factor_names[np.argmax(np.abs(beta_hat[1:]))]
        render_ib(
            f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">📊 Interpretation for {fund_name}:</span><br>'
            + steps_html([
                ("Model Fit",
                 txt_s(f'Adj R² = {adj_r2:.4f} → factors explain ') + hl(f"{adj_r2*100:.1f}%")
                 + txt_s(f' of fund return variance. F = {f_stat:.2f} (p={p_f:.4f}), model is ')
                 + (gt("statistically significant. ✓") if p_f<0.05 else rt2("not significant."))),
                ("Alpha (Skill)",
                 txt_s(f'Monthly α = {beta_hat[0]*100:.4f}% → Annualised = ') + hl(f"{alpha_ann:.4f}%")
                 + txt_s('. ') + (gt("Positive alpha — fund manager adds value! ✓")
                                   if beta_hat[0]>0 else rt2("Negative alpha — no outperformance after factor adjustment."))),
                ("Factor Exposures",
                 txt_s(f'Dominant factor: ') + hl(dominant) + txt_s(f'. ')
                 + txt_s(f'Market β = {beta_hat[1]:.4f} ({"aggressive" if beta_hat[1]>1 else "defensive"}), '
                         f'SMB β = {beta_hat[2]:.4f} ({"small-cap tilt" if beta_hat[2]>0 else "large-cap tilt"})')),
                ("Multicollinearity",
                 txt_s(f'Max VIF = {max(vifs):.2f}. ')
                 + (gt("No multicollinearity concern. ✓") if max(vifs)<5
                    else (org("Moderate multicollinearity — monitor.") if max(vifs)<10
                          else rt2("High multicollinearity — consider variable reduction!")))),
            ]),
            "gold"
        )


def _compute_vif(factors):
    vifs = []
    X_all = np.column_stack(factors)
    for i in range(len(factors)):
        y_i = X_all[:,i]
        x_i = np.column_stack([X_all[:,j] for j in range(len(factors)) if j!=i])
        X_i = np.column_stack([np.ones(len(y_i)), x_i])
        b   = np.linalg.lstsq(X_i, y_i, rcond=None)[0]
        yh  = X_i @ b
        ss  = np.sum((y_i-y_i.mean())**2)
        r2  = 1-np.sum((y_i-yh)**2)/ss if ss>0 else 0
        vifs.append(1/(1-r2) if r2<1 else 999)
    return vifs


def _mlr_plots(y_actual, y_hat, residuals, betas, ses, factor_names, fund_name):
    fig = plt.figure(figsize=(14, 10), facecolor="#0a1628")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Actual vs Fitted ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0,:2])
    ax1.set_facecolor("#112240")
    t = np.arange(len(y_actual))
    ax1.plot(t, y_actual,  color="#64ffda", lw=1.5, alpha=0.8, label="Actual")
    ax1.plot(t, y_hat,     color="#FFD700", lw=1.8, label="Fitted")
    ax1.fill_between(t, y_actual, y_hat, alpha=0.15, color="#dc3545")
    ax1.set_xlabel("Month", color="#8892b0", fontsize=9)
    ax1.set_ylabel("Excess Return (%)", color="#8892b0", fontsize=9)
    ax1.set_title(f"{fund_name} — Actual vs Fitted Returns", color="#FFD700", fontsize=11)
    ax1.legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")
    _style_ax(ax1)

    # ── Coefficient Plot ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0,2])
    ax2.set_facecolor("#112240")
    colors = ["#FFD700","#28a745","#ff9f43","#a29bfe"]
    y_pos  = np.arange(len(betas))
    bars   = ax2.barh(y_pos, betas, color=colors[:len(betas)], alpha=0.8, height=0.5)
    ax2.errorbar(np.array(betas), y_pos, xerr=1.96*np.array(ses), fmt='none', color="#e6f1ff", capsize=4, lw=1.5)
    ax2.axvline(0, color="#8892b0", lw=1, ls="--")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([n.split("(")[0].strip() for n in factor_names], color="#e6f1ff", fontsize=8)
    ax2.set_xlabel("Coefficient (β̂) ± 95% CI", color="#8892b0", fontsize=9)
    ax2.set_title("Factor Loadings", color="#FFD700", fontsize=11)
    _style_ax(ax2)

    # ── Residuals vs Fitted ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1,0])
    ax3.set_facecolor("#112240")
    ax3.scatter(y_hat, residuals, color="#ADD8E6", alpha=0.6, s=35)
    ax3.axhline(0, color="#FFD700", lw=1.5, ls="--")
    ax3.set_xlabel("Fitted Values (%)", color="#8892b0", fontsize=9)
    ax3.set_ylabel("Residuals (%)", color="#8892b0", fontsize=9)
    ax3.set_title("Residuals vs Fitted", color="#FFD700", fontsize=11)
    _style_ax(ax3)

    # ── Residual Histogram ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1,1])
    ax4.set_facecolor("#112240")
    ax4.hist(residuals, bins=20, color="#004d80", edgecolor="#ADD8E6", alpha=0.8)
    mu, sg = np.mean(residuals), np.std(residuals)
    xn = np.linspace(mu-4*sg, mu+4*sg, 200)
    ax4_t = ax4.twinx()
    ax4_t.plot(xn, stats.norm.pdf(xn,mu,sg), color="#FFD700", lw=2)
    ax4_t.set_yticks([]); ax4_t.set_facecolor("#112240")
    ax4.set_xlabel("Residuals (%)", color="#8892b0", fontsize=9)
    ax4.set_ylabel("Frequency", color="#8892b0", fontsize=9)
    ax4.set_title("Residual Distribution", color="#FFD700", fontsize=11)
    _style_ax(ax4)

    # ── Q-Q Plot ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1,2])
    ax5.set_facecolor("#112240")
    osm, osr = stats.probplot(residuals, dist="norm")
    ax5.scatter(osm[0], osm[1], color="#64ffda", alpha=0.7, s=35)
    ax5.plot(osm[0], osm[0]*osr[0]+osr[1], color="#FFD700", lw=2)
    ax5.set_xlabel("Theoretical Quantiles", color="#8892b0", fontsize=9)
    ax5.set_ylabel("Sample Quantiles (%)", color="#8892b0", fontsize=9)
    ax5.set_title("Q-Q Plot (Normality)", color="#FFD700", fontsize=11)
    _style_ax(ax5)

    return fig


def _style_ax(ax):
    ax.tick_params(colors="#8892b0", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#1e3a5f")
    ax.grid(color="#1e3a5f", alpha=0.3, lw=0.5)
