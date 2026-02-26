"""
tab_code.py — Python code reference for regression
"""
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2,
    lb_t, txt_s, p, two_col, table_html, metric_row,
    section_heading, S, FH, FB, FM, TXT, NO_SEL
)

SLR_CODE = '''import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ============================================================
# SIMPLE LINEAR REGRESSION — CAPM Beta Estimation
# ============================================================

# 1. Manual OLS (from scratch)
def ols_slr(x, y):
    n     = len(x)
    x_bar, y_bar = x.mean(), y.mean()
    b1    = np.sum((x-x_bar)*(y-y_bar)) / np.sum((x-x_bar)**2)
    b0    = y_bar - b1*x_bar
    y_hat = b0 + b1*x
    res   = y - y_hat
    sse   = np.sum(res**2);  sst = np.sum((y-y_bar)**2)
    r2    = 1 - sse/sst;     mse = sse/(n-2)
    se_b1 = np.sqrt(mse / np.sum((x-x_bar)**2))
    t_b1  = b1/se_b1;        p_b1 = 2*(1-stats.t.cdf(abs(t_b1), df=n-2))
    return {"b0":b0, "b1":b1, "r2":r2, "t":t_b1, "p":p_b1, "se":se_b1}

# 2. Using statsmodels (recommended in practice)
def capm_regression(stock_excess, market_excess):
    X = sm.add_constant(market_excess)         # adds intercept column
    model = sm.OLS(stock_excess, X).fit()
    print(model.summary())
    alpha = model.params[0]                     # Jensen's alpha
    beta  = model.params[1]                     # Systematic risk (CAPM beta)
    r2    = model.rsquared
    return model, alpha, beta

# 3. Simulate CAPM data and run regression
np.random.seed(42)
n          = 60                                 # 5 years monthly
rf         = 0.065/12                           # monthly risk-free rate
mkt_excess = np.random.normal(0.006, 0.045, n)
true_alpha = 0.002; true_beta = 1.25
stock_excess = true_alpha + true_beta*mkt_excess + np.random.normal(0, 0.03, n)

model, alpha, beta = capm_regression(stock_excess, mkt_excess)
print(f"Alpha (monthly): {alpha:.4f}  ({alpha*12*100:.2f}% annualized)")
print(f"Beta:            {beta:.4f}")
print(f"R²:              {model.rsquared:.4f}")

# 4. 95% Confidence Interval for Beta
conf = model.conf_int(alpha=0.05)
print(f"Beta 95% CI: [{conf.iloc[1,0]:.4f}, {conf.iloc[1,1]:.4f}]")

# 5. Prediction with prediction interval
mkt_new = np.array([0.01, 0.02, -0.01])
X_new   = sm.add_constant(mkt_new)
pred    = model.get_prediction(X_new)
print(pred.summary_frame(alpha=0.05))
'''

MLR_CODE = '''# ============================================================
# MULTIPLE LINEAR REGRESSION — Fama-French 3-Factor Model
# ============================================================
import numpy as np
import statsmodels.api as sm
import pandas as pd

def fama_french_regression(fund_returns, mkt, smb, hml, rf=0):
    # Excess returns
    fund_excess = fund_returns - rf
    X = sm.add_constant(np.column_stack([mkt, smb, hml]))
    X_df = pd.DataFrame(X, columns=["const","MKT","SMB","HML"])
    model = sm.OLS(fund_excess, X_df).fit()
    print(model.summary())
    return model

# Simulate Fama-French data
np.random.seed(7); n = 60
mkt = np.random.normal(0.006, 0.045, n)
smb = np.random.normal(0.002, 0.025, n)
hml = np.random.normal(0.002, 0.022, n)
true_alpha = 0.0018
fund_excess = true_alpha + 0.95*mkt + 0.40*smb + 0.30*hml + np.random.normal(0, 0.018, n)

model = fama_french_regression(fund_excess, mkt, smb, hml)

# Diagnose multicollinearity with VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_check = np.column_stack([mkt, smb, hml])
for i, name in enumerate(["MKT","SMB","HML"]):
    vif = variance_inflation_factor(X_check, i)
    print(f"VIF({name}) = {vif:.3f}")

# Model comparison: AIC/BIC
models = {}
for factors, names in [
    ([mkt],           "CAPM"),
    ([mkt,smb,hml],   "FF3"),
    ([mkt,smb,hml,np.random.normal(0,0.03,n)], "FF4"),
]:
    X = sm.add_constant(np.column_stack(factors))
    m = sm.OLS(fund_excess, X).fit()
    models[names] = {"AIC":m.aic, "BIC":m.bic, "Adj_R2":m.rsquared_adj}

print(pd.DataFrame(models).T)
'''

DIAG_CODE = '''# ============================================================
# REGRESSION DIAGNOSTICS
# ============================================================
import numpy as np
import scipy.stats as stats
import statsmodels.stats.diagnostic as diag
import statsmodels.stats.stattools as st_tools

def run_diagnostics(model, y, X):
    res = model.resid
    print("=" * 55)

    # 1. Normality — Jarque-Bera
    jb_stat, jb_p, skew, kurt = diag.jarque_bera(res)
    print(f"Jarque-Bera:  stat={jb_stat:.4f}, p={jb_p:.4f}", "✓" if jb_p>0.05 else "✗")
    print(f"  Skewness={skew:.4f}, Kurtosis={kurt:.4f}")

    # 2. Normality — Shapiro-Wilk (small samples)
    sw_stat, sw_p = stats.shapiro(res[:50])
    print(f"Shapiro-Wilk: stat={sw_stat:.4f}, p={sw_p:.4f}", "✓" if sw_p>0.05 else "✗")

    # 3. Autocorrelation — Durbin-Watson
    dw = st_tools.durbin_watson(res)
    print(f"Durbin-Watson: {dw:.4f}", "✓ No autocorr" if 1.5<dw<2.5 else "✗ Autocorrelation")

    # 4. Heteroscedasticity — Breusch-Pagan
    bp_stat, bp_p, f_stat, f_p = diag.het_breuschpagan(res, model.model.exog)
    print(f"Breusch-Pagan: stat={bp_stat:.4f}, p={bp_p:.4f}", "✓" if bp_p>0.05 else "✗ Heteroscedastic")

    # 5. RESET Test (misspecification)
    reset_stat, reset_p = diag.linear_reset(model, use_f=True)
    print(f"RESET Test:   stat={reset_stat:.4f}, p={reset_p:.4f}", "✓" if reset_p>0.05 else "✗ Misspecified")
    print("=" * 55)

# Remedies for violations
def robust_regression(y, X):
    import statsmodels.api as sm
    model = sm.OLS(y, sm.add_constant(X)).fit()

    # HC3 robust standard errors (heteroscedasticity)
    model_hc = model.get_robustcov_results(cov_type="HC3")

    # HAC / Newey-West (heteroscedasticity + autocorrelation)
    model_hac = model.get_robustcov_results(cov_type="HAC", maxlags=4)

    return model_hc, model_hac
'''


def tab_code():
    render_card("🐍 Python Implementation Guide",
        p(f'Complete Python code for SLR, MLR, and diagnostics using {hl("NumPy")}, '
          f'{hl("SciPy")}, and {hl("statsmodels")}.')
        + two_col(
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">📦 Required Libraries</span><br>'
               + fml("pip install numpy scipy statsmodels\n         pandas matplotlib seaborn"),
               "green"),
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">📚 Key Classes</span><br>'
               + p(f'{hl("sm.OLS")} — Ordinary Least Squares<br>'
                   f'{hl("sm.GLS")} — Generalised Least Squares<br>'
                   f'{hl("sm.WLS")} — Weighted Least Squares<br>'
                   f'{hl("sm.RLM")} — Robust Least Squares'),
               "gold"),
        )
    )

    topic = st.radio("Code Section",
                     ["SLR — CAPM Beta", "MLR — Fama-French", "Diagnostics"],
                     horizontal=True, key="code_topic")

    if topic == "SLR — CAPM Beta":
        st.code(SLR_CODE, language="python")
    elif topic == "MLR — Fama-French":
        st.code(MLR_CODE, language="python")
    else:
        st.code(DIAG_CODE, language="python")

    # Live runner
    render_card("▶ Live Calculator — Quick Regression",
        ib(f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff">'
           f'Enter two comma-separated series to run a quick OLS regression.</span>', "blue")
    )

    col1, col2 = st.columns(2)
    x_in = col1.text_input("X values (comma-separated)", value="2,4,6,8,10,12,14,16,18,20")
    y_in = col2.text_input("Y values (comma-separated)", value="3.2,7.1,10.5,13.8,18.1,21.3,25.9,28.4,33.1,37.2")

    if st.button("▶ Run Quick Regression", key="quick_reg"):
        try:
            x = np.array([float(v.strip()) for v in x_in.split(",")])
            y = np.array([float(v.strip()) for v in y_in.split(",")])
            assert len(x)==len(y), "X and Y must have same length"

            xb, yb = x.mean(), y.mean()
            b1 = np.sum((x-xb)*(y-yb))/np.sum((x-xb)**2); b0 = yb-b1*xb
            yh = b0+b1*x; res = y-yh
            n = len(x); sse=np.sum(res**2); sst=np.sum((y-yb)**2)
            r2=1-sse/sst; mse=sse/(n-2)
            se_b1=np.sqrt(mse/np.sum((x-xb)**2)); t1=b1/se_b1
            p1=2*(1-stats.t.cdf(abs(t1),df=n-2))
            corr=np.corrcoef(x,y)[0,1]

            metric_row([
                ("β̂₀ (Intercept)", f"{b0:.4f}", None),
                ("β̂₁ (Slope)",     f"{b1:.4f}", None),
                ("R²",              f"{r2:.4f}", None),
                ("Pearson r",       f"{corr:.4f}", None),
            ])
            metric_row([
                ("t-stat (β₁)",  f"{t1:.4f}", None),
                ("p-value",      f"{p1:.4f}", None),
                ("n",            f"{n}", None),
                ("RMSE",         f"{np.sqrt(mse):.4f}", None),
            ])

            fig, axes = plt.subplots(1,2, figsize=(12,4), facecolor="#0a1628")
            for ax in axes: ax.set_facecolor("#112240")
            axes[0].scatter(x,y, color="#64ffda", s=60, zorder=3, label="Data")
            xl=np.linspace(x.min(),x.max(),200)
            axes[0].plot(xl, b0+b1*xl, color="#FFD700", lw=2.5, label=f"y = {b0:.3f} + {b1:.4f}x")
            axes[0].set_xlabel("X", color="#8892b0"); axes[0].set_ylabel("Y", color="#8892b0")
            axes[0].set_title("Regression Line", color="#FFD700", fontsize=11)
            axes[0].legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=9, edgecolor="#1e3a5f")
            _sax(axes[0])
            axes[1].scatter(yh, res, color="#ADD8E6", s=50, alpha=0.8)
            axes[1].axhline(0, color="#FFD700", lw=1.5, ls="--")
            axes[1].set_xlabel("Fitted", color="#8892b0"); axes[1].set_ylabel("Residuals", color="#8892b0")
            axes[1].set_title("Residuals vs Fitted", color="#FFD700", fontsize=11)
            _sax(axes[1])
            plt.tight_layout(pad=1.5); st.pyplot(fig, use_container_width=True); plt.close(fig)

            # Quick interpretation
            sig_txt = gt("Significant ✓") if p1<0.05 else rt2("Not significant ✗")
            render_ib(
                f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Result:</span> '
                + txt_s(f'Ŷ = {b0:.4f} + {b1:.4f}X | R² = {r2:.4f} | ')
                + sig_txt + txt_s(f' (p = {p1:.4f})'),
                "gold"
            )
        except Exception as e:
            render_ib(rt2(f"Error: {str(e)} — Check input format"), "red")

    section_heading("📊 Critical Values & Model Selection Reference")
    rows = []
    for k in [1,2,3,4,5]:
        f_c = stats.f.ppf(0.95, dfn=k, dfd=50)
        rows.append([txt_s(str(k)), txt_s(f"{f_c:.3f}"), txt_s(f"{k+1}"),
                     txt_s("Adj R² ↑ if F > 1"),
                     bdg("Include","green") if f_c<4 else bdg("Check","orange")])
    st.html(table_html(["k (predictors)","F_crit (α=5%,df₂=50)","Parameters","Rule of thumb","Decision"], rows))


def _sax(ax):
    ax.tick_params(colors="#8892b0", labelsize=8)
    for sp in ax.spines.values(): sp.set_color("#1e3a5f")
    ax.grid(color="#1e3a5f", alpha=0.3, lw=0.5)
