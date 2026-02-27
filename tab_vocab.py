"""
tab_vocab.py — Vocabulary & Concepts Hub
Education Hub with concept cards, glossary, formula sheet, cheat sheets
Matching the badge+card design shown in the screenshot
"""
import streamlit as st
from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, txt_s, p, steps_html, two_col, three_col, four_col,
    table_html, section_heading, S, FH, FB, FM, TXT, NO_SEL
)

# ── Local helpers ─────────────────────────────────────────────────
def _f(t):
    return (f'<span style="font-family:{FM};font-size:.83rem;color:#64ffda;'
            f'-webkit-text-fill-color:#64ffda">{t}</span>')

def concept_card(icon, title, title_color, border_color, bg_color, items):
    """
    Replicates the card style in the screenshot:
    colored border-left, icon+title header, badge+text rows
    """
    rows_html = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:10px;'
        f'margin-bottom:9px;line-height:1.55;{NO_SEL}">'
        f'{item["badge"]}'
        f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;'
        f'font-family:{FB};font-size:.88rem">{item["text"]}</span>'
        f'</div>'
        for item in items
    )
    return (
        f'<div style="background:{bg_color};border-left:4px solid {border_color};'
        f'border-radius:10px;padding:18px 18px 14px;height:100%;'
        f'user-select:none;-webkit-user-select:none">'
        f'<div style="font-family:{FH};font-size:1.05rem;color:{title_color};'
        f'-webkit-text-fill-color:{title_color};font-weight:700;margin-bottom:13px">'
        f'{icon} {title}</div>'
        f'{rows_html}'
        f'</div>'
    )

def term_card(term, symbol, definition, formula=None, example=None,
              badge_label=None, badge_variant="blue", finance_note=None):
    """Full glossary term card with optional formula + example."""
    sym_html = (f'<span style="font-family:{FM};font-size:.82rem;color:#64ffda;'
                f'-webkit-text-fill-color:#64ffda;margin-left:8px">{symbol}</span>') if symbol else ""
    bdg_html = bdg(badge_label, badge_variant) if badge_label else ""
    formula_html = fml(formula) if formula else ""
    example_html = (
        f'<div style="background:rgba(255,215,0,0.08);border-left:3px solid #FFD700;'
        f'border-radius:5px;padding:9px 12px;margin-top:8px;'
        f'color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;'
        f'font-family:{FB};font-size:.85rem;line-height:1.6">'
        f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Example: </span>'
        f'{example}</div>'
    ) if example else ""
    finance_html = (
        f'<div style="margin-top:8px;color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;'
        f'font-family:{FB};font-size:.84rem">'
        f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">📈 Finance: </span>'
        f'{finance_note}</div>'
    ) if finance_note else ""

    return (
        f'<div style="background:#112240;border:1px solid #1e3a5f;border-radius:10px;'
        f'padding:16px 18px;margin-bottom:14px;{NO_SEL}">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
        f'<span style="font-family:{FH};font-size:1.02rem;color:#FFD700;'
        f'-webkit-text-fill-color:#FFD700;font-weight:700">{term}</span>'
        f'{sym_html}{bdg_html}</div>'
        f'<div style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;'
        f'font-family:{FB};font-size:.9rem;line-height:1.65;margin-bottom:6px">'
        f'{definition}</div>'
        f'{formula_html}{example_html}{finance_html}'
        f'</div>'
    )

def mini_card(title, color, content_html):
    """Compact card for cheat sheet / quick reference items."""
    return (
        f'<div style="background:rgba(0,51,102,0.45);border:1px solid {color};'
        f'border-radius:8px;padding:14px 15px;{NO_SEL}">'
        f'<div style="color:{color};-webkit-text-fill-color:{color};'
        f'font-family:{FH};font-size:.95rem;font-weight:700;margin-bottom:10px">{title}</div>'
        f'{content_html}'
        f'</div>'
    )

def _row(label, value):
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:4px 0;border-bottom:1px solid rgba(30,58,95,0.5);{NO_SEL}">'
        f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0;'
        f'font-family:{FB};font-size:.84rem">{label}</span>'
        f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;'
        f'font-family:{FM};font-size:.84rem">{value}</span>'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════
# SECTION DATA
# ══════════════════════════════════════════════════════════════════

CONCEPT_CARDS = {
    "Core OLS": [
        {
            "icon": "📐", "title": "OLS Fundamentals", "title_color": "#FFD700",
            "border_color": "#FFD700", "bg_color": "rgba(255,215,0,0.07)",
            "items": [
                {"badge": bdg("Minimise SSE","gold"),   "text": "Σ(Yᵢ − Ŷᵢ)² → minimum"},
                {"badge": bdg("BLUE","blue"),            "text": "Best Linear Unbiased Estimator"},
                {"badge": bdg("Normal equations","blue"),"text": "XᵀXβ̂ = XᵀY solved analytically"},
                {"badge": bdg("Gauss-Markov","green"),   "text": "OLS is BLUE under CLRM assumptions"},
                {"badge": bdg("Fitted values","purple"), "text": "Ŷ = β̂₀ + β̂₁X (projection onto X space)"},
                {"badge": bdg("Residuals","orange"),     "text": "e = Y − Ŷ; note Σeᵢ = 0 always"},
            ]
        },
        {
            "icon": "📊", "title": "Decomposition of Variance", "title_color": "#ADD8E6",
            "border_color": "#ADD8E6", "bg_color": "rgba(0,51,102,0.5)",
            "items": [
                {"badge": bdg("SST","blue"),    "text": "Total Sum of Squares = Σ(Yᵢ−Ȳ)²"},
                {"badge": bdg("SSR","green"),   "text": "Regression SS = Σ(Ŷᵢ−Ȳ)² [Explained]"},
                {"badge": bdg("SSE","red"),     "text": "Error SS = Σ(Yᵢ−Ŷᵢ)² [Unexplained]"},
                {"badge": bdg("Identity","gold"),"text": "SST = SSR + SSE always"},
                {"badge": bdg("R²","gold"),     "text": "SSR/SST = 1 − SSE/SST"},
                {"badge": bdg("MSE","orange"),  "text": "SSE/(n−k−1) = unbiased σ̂² estimator"},
            ]
        },
        {
            "icon": "🔢", "title": "Coefficient Interpretation", "title_color": "#28a745",
            "border_color": "#28a745", "bg_color": "rgba(40,167,69,0.08)",
            "items": [
                {"badge": bdg("β̂₀ Intercept","blue"),   "text": "E(Y) when all X = 0 (may lack economic meaning)"},
                {"badge": bdg("β̂₁ Slope","gold"),       "text": "ΔY per unit ΔX, holding all other X constant"},
                {"badge": bdg("Ceteris paribus","purple"),"text": "Each β isolates its variable's marginal effect"},
                {"badge": bdg("Elasticity","orange"),    "text": "Log-log model: β = % change Y / % change X"},
                {"badge": bdg("Dummy variable","green"), "text": "Binary 0/1 X — captures group differences"},
                {"badge": bdg("Interaction term","red"), "text": "X₁×X₂ — slope of X₁ varies with X₂"},
            ]
        },
    ],
    "Hypothesis Testing": [
        {
            "icon": "🧪", "title": "t-Test on Coefficients", "title_color": "#FFD700",
            "border_color": "#FFD700", "bg_color": "rgba(255,215,0,0.07)",
            "items": [
                {"badge": bdg("H₀","blue"),        "text": "β_j = 0 (variable has no effect)"},
                {"badge": bdg("t-stat","gold"),    "text": "t = β̂_j / SE(β̂_j) ~ t(n−k−1)"},
                {"badge": bdg("Reject","red"),     "text": "|t| > t_crit → variable is significant"},
                {"badge": bdg("p-value","purple"), "text": "P(|T| > |t|) → probability of Type I error"},
                {"badge": bdg("95% CI","green"),   "text": "β̂_j ± t_crit × SE(β̂_j)"},
                {"badge": bdg("Two-tailed","blue"),"text": "Used when direction of β is not pre-specified"},
            ]
        },
        {
            "icon": "📋", "title": "F-Test (Overall Significance)", "title_color": "#ADD8E6",
            "border_color": "#ADD8E6", "bg_color": "rgba(0,51,102,0.5)",
            "items": [
                {"badge": bdg("H₀","blue"),       "text": "β₁ = β₂ = ... = βₖ = 0 (no predictor matters)"},
                {"badge": bdg("F-stat","gold"),   "text": "F = (R²/k) / ((1−R²)/(n−k−1)) = MSR/MSE"},
                {"badge": bdg("df₁ = k","orange"),"text": "Numerator df = number of restrictions"},
                {"badge": bdg("df₂","orange"),    "text": "Denominator df = n − k − 1"},
                {"badge": bdg("Reject","red"),    "text": "F > F_crit → at least one β_j ≠ 0"},
                {"badge": bdg("SLR link","green"),"text": "In SLR: F = t² (exactly equivalent)"},
            ]
        },
        {
            "icon": "⚠", "title": "Type I & Type II Errors", "title_color": "#dc3545",
            "border_color": "#dc3545", "bg_color": "rgba(220,53,69,0.08)",
            "items": [
                {"badge": bdg("Type I Error","red"),      "text": "Reject H₀ when it is true = α (false positive)"},
                {"badge": bdg("Type II Error","orange"),  "text": "Fail to reject false H₀ = β (false negative)"},
                {"badge": bdg("Power = 1−β","green"),     "text": "Probability of correctly rejecting false H₀"},
                {"badge": bdg("Significance α","gold"),   "text": "Chosen threshold: 1%, 5%, or 10%"},
                {"badge": bdg("Trade-off","purple"),      "text": "Lowering α reduces Type I but increases Type II"},
                {"badge": bdg("Finance impact","blue"),   "text": "Type I: invest in bad strategy; Type II: miss good one"},
            ]
        },
    ],
    "CLRM Assumptions": [
        {
            "icon": "✅", "title": "Normality of Residuals", "title_color": "#28a745",
            "border_color": "#28a745", "bg_color": "rgba(40,167,69,0.08)",
            "items": [
                {"badge": bdg("Jarque-Bera","gold"),   "text": "Tests skewness + kurtosis; JB ~ χ²(2)"},
                {"badge": bdg("Shapiro-Wilk","blue"),  "text": "More powerful for n < 50"},
                {"badge": bdg("Q-Q Plot","purple"),    "text": "Visual check — points lie on 45° line if normal"},
                {"badge": bdg("Fat tails","orange"),   "text": "K > 3 (leptokurtic) — very common in finance"},
                {"badge": bdg("Remedy","red"),         "text": "Robust SE, bootstrap CI, or transform Y"},
                {"badge": bdg("H₀","blue"),            "text": "Residuals are normally distributed"},
            ]
        },
        {
            "icon": "📈", "title": "Heteroscedasticity", "title_color": "#dc3545",
            "border_color": "#dc3545", "bg_color": "rgba(220,53,69,0.08)",
            "items": [
                {"badge": bdg("Breusch-Pagan","red"),    "text": "Regress ε² on X; LM = n·R²_aux ~ χ²(k)"},
                {"badge": bdg("White Test","orange"),    "text": "Also includes squared and cross terms"},
                {"badge": bdg("Scale-Location","blue"),  "text": "√|eᵢ| vs Ŷᵢ — should be flat"},
                {"badge": bdg("Effect","purple"),        "text": "β̂ unbiased but SE biased → invalid t/F tests"},
                {"badge": bdg("HC3 Robust SE","gold"),   "text": "White's correction — valid inference"},
                {"badge": bdg("H₀","blue"),              "text": "Var(εᵢ) = σ² (constant variance)"},
            ]
        },
        {
            "icon": "🔄", "title": "Autocorrelation", "title_color": "#ff9f43",
            "border_color": "#ff9f43", "bg_color": "rgba(255,159,67,0.08)",
            "items": [
                {"badge": bdg("Durbin-Watson","orange"), "text": "d ≈ 2 = no autocorr; < 2 = positive"},
                {"badge": bdg("Breusch-Godfrey","blue"), "text": "Higher-order test; works with lagged Y"},
                {"badge": bdg("ACF Plot","green"),       "text": "Autocorrelation Function — spikes signal AR"},
                {"badge": bdg("Effect","purple"),        "text": "β̂ unbiased but SE biased — t-tests invalid"},
                {"badge": bdg("Newey-West","gold"),      "text": "HAC robust SE — corrects for autocorrelation"},
                {"badge": bdg("H₀","blue"),              "text": "Cov(εᵢ, εⱼ) = 0 for all i ≠ j"},
            ]
        },
        {
            "icon": "🔗", "title": "Multicollinearity", "title_color": "#a29bfe",
            "border_color": "#a29bfe", "bg_color": "rgba(162,155,254,0.08)",
            "items": [
                {"badge": bdg("VIF","purple"),           "text": "1/(1−Rⱼ²); > 10 = serious problem"},
                {"badge": bdg("Correlation matrix","blue"),"text": "Pairwise check — |r| > 0.8 signals risk"},
                {"badge": bdg("Effect","red"),           "text": "β̂ unbiased but unstable; SE inflated"},
                {"badge": bdg("Ridge regression","orange"),"text": "Adds λΣβ² penalty; shrinks correlated β̂"},
                {"badge": bdg("PCA","gold"),             "text": "Orthogonal components eliminate collinearity"},
                {"badge": bdg("H₀ (perfect)","blue"),   "text": "No exact linear relationship among X's"},
            ]
        },
    ],
    "Finance Models": [
        {
            "icon": "💹", "title": "CAPM / SCL", "title_color": "#FFD700",
            "border_color": "#FFD700", "bg_color": "rgba(255,215,0,0.07)",
            "items": [
                {"badge": bdg("SCL","blue"),           "text": "Rᵢ−Rf = α + β(Rₘ−Rf) + ε"},
                {"badge": bdg("Alpha α","gold"),       "text": "Jensen's α — excess risk-adjusted return"},
                {"badge": bdg("Beta β","orange"),      "text": "Systematic risk; β=Cov(Rᵢ,Rₘ)/Var(Rₘ)"},
                {"badge": bdg("β > 1","red"),          "text": "Aggressive — amplifies market moves"},
                {"badge": bdg("β < 1","green"),        "text": "Defensive — dampens market moves"},
                {"badge": bdg("1 − R²","purple"),      "text": "Idiosyncratic (unsystematic) risk share"},
            ]
        },
        {
            "icon": "🏦", "title": "Fama-French 3-Factor", "title_color": "#ADD8E6",
            "border_color": "#ADD8E6", "bg_color": "rgba(0,51,102,0.5)",
            "items": [
                {"badge": bdg("Market β","blue"),  "text": "Rₘ−Rf: systematic market risk premium"},
                {"badge": bdg("SMB β","gold"),     "text": "Small Minus Big: +ve = small-cap tilt"},
                {"badge": bdg("HML β","green"),    "text": "High Minus Low: +ve = value tilt"},
                {"badge": bdg("WML β","purple"),   "text": "Carhart momentum: +ve = winners tilt"},
                {"badge": bdg("Alpha","orange"),   "text": "Unexplained excess return (manager skill)"},
                {"badge": bdg("R² vs CAPM","red"), "text": "FF3 typically has higher R² than CAPM"},
            ]
        },
        {
            "icon": "📐", "title": "Bond Yield Model", "title_color": "#28a745",
            "border_color": "#28a745", "bg_color": "rgba(40,167,69,0.08)",
            "items": [
                {"badge": bdg("Model","blue"),         "text": "Yield = β₀ + β₁(Duration) + ε"},
                {"badge": bdg("β₀ intercept","gold"),  "text": "Approximate risk-free (base) rate"},
                {"badge": bdg("β₁ slope","green"),     "text": "Term premium per year of duration (bps)"},
                {"badge": bdg("Normal curve","blue"),   "text": "β₁ > 0: upward sloping yield curve"},
                {"badge": bdg("Inverted","red"),        "text": "β₁ < 0: short rates > long rates"},
                {"badge": bdg("Convexity","purple"),   "text": "Add Duration² for non-linear term structure"},
            ]
        },
        {
            "icon": "💳", "title": "Credit Risk Regression", "title_color": "#dc3545",
            "border_color": "#dc3545", "bg_color": "rgba(220,53,69,0.08)",
            "items": [
                {"badge": bdg("PD model","red"),       "text": "PD = β₀ + β₁(D/E) + β₂(ICR) + β₃(CR) + ε"},
                {"badge": bdg("D/E ratio +","red"),    "text": "Higher leverage → higher PD (positive β)"},
                {"badge": bdg("ICR −","green"),        "text": "Higher interest coverage → lower PD (negative β)"},
                {"badge": bdg("Altman Z","gold"),      "text": "Classic 5-variable bankruptcy prediction model"},
                {"badge": bdg("Logit/Probit","purple"),"text": "Better for binary default outcomes (0/1)"},
                {"badge": bdg("LGD","orange"),         "text": "Loss Given Default — separate regression needed"},
            ]
        },
    ],
    "Model Selection": [
        {
            "icon": "📊", "title": "Information Criteria", "title_color": "#FFD700",
            "border_color": "#FFD700", "bg_color": "rgba(255,215,0,0.07)",
            "items": [
                {"badge": bdg("AIC","gold"),    "text": "2k − 2ln(L); lower = better; penalises parameters"},
                {"badge": bdg("BIC","orange"),  "text": "k·ln(n) − 2ln(L); heavier k penalty than AIC"},
                {"badge": bdg("BIC vs AIC","purple"),"text": "BIC preferred for model selection; AIC for prediction"},
                {"badge": bdg("Adj R²","blue"), "text": "Increases only if new variable adds real explanatory power"},
                {"badge": bdg("RMSE","green"),  "text": "√MSE — in Y units; useful for prediction accuracy"},
                {"badge": bdg("Mallows Cₚ","red"),"text": "Cₚ ≈ k+1 indicates good fit; < k+1 preferred"},
            ]
        },
        {
            "icon": "⚙", "title": "Variable Selection", "title_color": "#ADD8E6",
            "border_color": "#ADD8E6", "bg_color": "rgba(0,51,102,0.5)",
            "items": [
                {"badge": bdg("Forward","green"),      "text": "Start empty; add most significant variable each step"},
                {"badge": bdg("Backward","orange"),    "text": "Start full; remove least significant each step"},
                {"badge": bdg("Stepwise","blue"),      "text": "Combines forward + backward; can add/remove"},
                {"badge": bdg("LASSO","purple"),       "text": "L1 penalty λΣ|β|; forces some β → exactly 0"},
                {"badge": bdg("Ridge","gold"),         "text": "L2 penalty λΣβ²; shrinks but never zeros"},
                {"badge": bdg("Elastic Net","red"),    "text": "Combines L1 + L2; balances LASSO and Ridge"},
            ]
        },
        {
            "icon": "🔍", "title": "Specification Tests", "title_color": "#a29bfe",
            "border_color": "#a29bfe", "bg_color": "rgba(162,155,254,0.08)",
            "items": [
                {"badge": bdg("RESET","purple"),     "text": "Ramsey RESET: tests functional form misspecification"},
                {"badge": bdg("Chow test","gold"),   "text": "Tests structural break (stability) across two groups"},
                {"badge": bdg("CUSUM","blue"),       "text": "Tests parameter stability over time (time series)"},
                {"badge": bdg("Hausman","orange"),   "text": "OLS vs IV — tests for endogeneity in model"},
                {"badge": bdg("Omitted var","red"),  "text": "Missing X correlated with included X → biased β̂"},
                {"badge": bdg("Overfitting","green"),"text": "High R² but poor out-of-sample prediction"},
            ]
        },
    ],
}

GLOSSARY_TERMS = [
    {
        "term": "OLS (Ordinary Least Squares)",
        "symbol": "β̂ = (XᵀX)⁻¹Xᵀy",
        "definition": "A method of estimating regression coefficients by minimising the sum of squared residuals. Under CLRM assumptions, OLS is BLUE — Best Linear Unbiased Estimator.",
        "formula": "Minimise: Σ(Yᵢ − β̂₀ − β̂₁Xᵢ)²\nSolution: β̂₁ = Cov(X,Y)/Var(X)  |  β̂₀ = Ȳ − β̂₁X̄",
        "example": "Estimating CAPM beta by regressing Infosys excess returns on Nifty excess returns across 60 months.",
        "badge_label": "SLR / MLR", "badge_variant": "blue",
        "finance_note": "Most common regression technique in quantitative finance — CAPM, factor models, credit scoring.",
    },
    {
        "term": "R² — Coefficient of Determination",
        "symbol": "R² = SSR/SST",
        "definition": "Proportion of the dependent variable's total variance explained by the regression model. Range: 0 to 1. In SLR, R² = r² (square of Pearson correlation).",
        "formula": "R² = SSR/SST = 1 − SSE/SST\nAdj R² = 1 − (1−R²)(n−1)/(n−k−1)",
        "example": "CAPM regression gives R² = 0.72 → market explains 72% of stock return variation; 28% is idiosyncratic.",
        "badge_label": "Goodness of Fit", "badge_variant": "gold",
        "finance_note": "In CAPM: R² = systematic risk share; 1−R² = unsystematic (diversifiable) risk share.",
    },
    {
        "term": "Beta (β) — Systematic Risk",
        "symbol": "β = Cov(Rᵢ,Rₘ)/Var(Rₘ)",
        "definition": "The slope coefficient in CAPM regression measuring a stock's sensitivity to market movements. β > 1: aggressive (amplifies market). β < 1: defensive (dampens market).",
        "formula": "SCL: Rᵢ−Rf = α + β(Rₘ−Rf) + ε\nβ = Σ(Xᵢ−X̄)(Yᵢ−Ȳ) / Σ(Xᵢ−X̄)²",
        "example": "Reliance Industries β = 1.18 → for every 1% Nifty move, Reliance expected to move 1.18%.",
        "badge_label": "CAPM", "badge_variant": "orange",
        "finance_note": "Estimated via OLS. Beta is time-varying — rolling regressions over 36 or 60 months are common.",
    },
    {
        "term": "Jensen's Alpha (α)",
        "symbol": "α = β̂₀ from SCL",
        "definition": "The intercept in the Security Characteristic Line (CAPM) regression. Represents risk-adjusted excess return. α > 0 indicates manager skill; α < 0 indicates underperformance.",
        "formula": "α = R̄fund − [Rf + β(R̄market − Rf)]",
        "example": "A fund with α = +0.25% monthly = +3% annualised alpha generates value beyond its market risk exposure.",
        "badge_label": "Active Management", "badge_variant": "green",
        "finance_note": "CFA exams frequently test alpha interpretation. Statistically: test H₀: α=0 using t-test.",
    },
    {
        "term": "BLUE — Best Linear Unbiased Estimator",
        "symbol": "Gauss-Markov Theorem",
        "definition": "OLS estimators have minimum variance among all linear unbiased estimators when CLRM assumptions hold. If any assumption is violated, OLS may no longer be best or unbiased.",
        "formula": "Conditions: Linearity, Exogeneity, Homoscedasticity,\nNo autocorrelation, No perfect multicollinearity",
        "example": "When heteroscedasticity is present, WLS (Weighted Least Squares) becomes more efficient than OLS.",
        "badge_label": "CLRM", "badge_variant": "purple",
        "finance_note": "BLUE property justifies using OLS for CAPM beta estimation under standard assumptions.",
    },
    {
        "term": "VIF — Variance Inflation Factor",
        "symbol": "VIF_j = 1/(1−R²_j)",
        "definition": "Measures how much the variance of β̂_j is inflated due to correlation with other predictors. R²_j is obtained by regressing Xj on all other X variables.",
        "formula": "VIF < 5: OK | VIF 5–10: Moderate | VIF > 10: Serious\nSE inflation = √VIF × SE(no collinearity)",
        "example": "In Fama-French model: VIF(MKT)=1.08, VIF(SMB)=1.12, VIF(HML)=1.15 — all acceptable.",
        "badge_label": "Multicollinearity", "badge_variant": "red",
        "finance_note": "Including both Nifty50 and Nifty500 as regressors creates VIF >> 10 — perfectly collinear.",
    },
    {
        "term": "Durbin-Watson Statistic",
        "symbol": "DW = Σ(eₜ−eₜ₋₁)²/Σeₜ²",
        "definition": "Tests for first-order autocorrelation in regression residuals. Range: 0 to 4. DW≈2 indicates no autocorrelation.",
        "formula": "DW ≈ 2(1−ρ) where ρ = autocorrelation of residuals\nDW < 1.5: Positive autocorr | DW > 2.5: Negative autocorr",
        "example": "DW = 0.92 in a monthly bond yield regression → strong positive autocorrelation; use Newey-West SE.",
        "badge_label": "Autocorrelation", "badge_variant": "orange",
        "finance_note": "Very common in financial time series due to momentum and mean-reversion effects.",
    },
    {
        "term": "Heteroscedasticity",
        "symbol": "Var(εᵢ) ≠ σ²",
        "definition": "When the variance of regression residuals is not constant across observations. Violates CLRM Assumption 3. Does NOT bias β̂ but makes standard errors incorrect.",
        "formula": "Breusch-Pagan: LM = n × R²_aux ~ χ²(k)\nRemedy: HC3 robust SE or WLS",
        "example": "In equity return regressions, residuals are larger during high-volatility periods (ARCH effects).",
        "badge_label": "CLRM Violation", "badge_variant": "red",
        "finance_note": "Near-universal in financial time series. Always test with Breusch-Pagan before reporting results.",
    },
    {
        "term": "Adjusted R²",
        "symbol": "Adj R² = 1−(1−R²)(n−1)/(n−k−1)",
        "definition": "Modified R² that penalises model complexity. Unlike R², Adj R² decreases when adding variables that do not improve fit. Used for comparing models with different numbers of predictors.",
        "formula": "Adj R² = 1 − (SSE/(n−k−1)) / (SST/(n−1))\nRange: can be negative if model is very poor",
        "example": "Model A (k=3): R²=0.72, Adj R²=0.70. Model B (k=5): R²=0.73, Adj R²=0.68. Prefer Model A.",
        "badge_label": "Model Comparison", "badge_variant": "gold",
        "finance_note": "Always use Adj R² in MLR. In Fama-French, adding a 4th factor should improve Adj R² to be retained.",
    },
    {
        "term": "p-value",
        "symbol": "P(|T| > |t_obs|)",
        "definition": "Probability of observing a test statistic as extreme as the computed one, assuming H₀ is true. Low p-value → evidence against H₀. Not the probability H₀ is true.",
        "formula": "Reject H₀ if p-value < α (significance level)\np-value = 2 × P(T > |t_obs|) for two-tailed test",
        "example": "p(β_CAPM) = 0.002 < 0.05 → reject H₀: β=0 → market exposure is statistically significant.",
        "badge_label": "Inference", "badge_variant": "blue",
        "finance_note": "FRM/CFA examiners frequently test p-value interpretation — it is NOT P(H₀ is true).",
    },
]

FORMULA_SECTIONS = {
    "SLR Formulas": [
        ("Model",              "Y = β₀ + β₁X + ε"),
        ("OLS Slope β̂₁",     "Σ(Xᵢ−X̄)(Yᵢ−Ȳ) / Σ(Xᵢ−X̄)²"),
        ("OLS Intercept β̂₀", "Ȳ − β̂₁X̄"),
        ("SE(β̂₁)",            "√(MSE / Σ(Xᵢ−X̄)²)"),
        ("t-statistic",        "β̂₁ / SE(β̂₁) ~ t(n−2)"),
        ("R²",                 "1 − SSE/SST = SSR/SST"),
        ("r (Pearson)",        "±√R²  [sign = sign of β̂₁]"),
    ],
    "MLR Formulas": [
        ("Matrix form",        "Y = Xβ + ε"),
        ("OLS solution",       "β̂ = (XᵀX)⁻¹Xᵀy"),
        ("Variance of β̂",     "σ²(XᵀX)⁻¹"),
        ("Adj R²",             "1 − (1−R²)(n−1)/(n−k−1)"),
        ("F-statistic",        "(R²/k) / ((1−R²)/(n−k−1))"),
        ("AIC",                "2k − 2ln(L)"),
        ("BIC",                "k·ln(n) − 2ln(L)"),
    ],
    "Diagnostic Tests": [
        ("Jarque-Bera",        "n(S²/6 + (K−3)²/24) ~ χ²(2)"),
        ("Durbin-Watson",      "Σ(eₜ−eₜ₋₁)² / Σeₜ²  [ideal ≈ 2]"),
        ("Breusch-Pagan",      "n × R²_aux ~ χ²(k)"),
        ("VIF",                "1 / (1 − R²_j)"),
        ("95% CI for β̂_j",    "β̂_j ± t_crit × SE(β̂_j)"),
        ("Power of test",      "1 − P(Type II Error) = 1 − β"),
    ],
    "Finance Models": [
        ("CAPM SCL",           "Rᵢ−Rf = α + β(Rₘ−Rf) + ε"),
        ("Beta",               "Cov(Rᵢ,Rₘ) / Var(Rₘ)"),
        ("Fama-French 3F",     "Rᵢ−Rf = α + β₁MKT + β₂SMB + β₃HML + ε"),
        ("Bond yield model",   "YTM = β₀ + β₁(Duration) + ε"),
        ("Credit risk",        "PD = β₀ + β₁(D/E) + β₂(ICR) + ... + ε"),
        ("P/E valuation",      "P/E = β₀ + β₁(ROE) + β₂(g) − β₃(Beta) + ε"),
    ],
}


# ══════════════════════════════════════════════════════════════════
# MAIN TAB
# ══════════════════════════════════════════════════════════════════

def tab_vocab():
    # Header
    render_card("📚 Education Hub — Vocabulary & Concept Reference",
        p(f'A complete visual reference for {hl("linear regression")} concepts, '
          f'sorted by topic. Use as a study companion alongside the calculator tabs.')
        + four_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">🃏 Concept Cards</span><br>'
               + p(f'{bdg(f"{sum(len(v) for v in CONCEPT_CARDS.values())} cards","gold")} across 5 themes<br>Badge-style visual layout'), "gold"),
            ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">📖 Glossary</span><br>'
               + p(f'{bdg(f"{len(GLOSSARY_TERMS)} key terms","blue")} with definitions<br>Formulas + finance examples'), "blue"),
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">📐 Formula Sheet</span><br>'
               + p(f'{bdg("4 sections","green")} SLR, MLR, Tests<br>Quick reference format'), "green"),
            ib(f'<span style="color:#a29bfe;-webkit-text-fill-color:#a29bfe;font-weight:600">🗺 Concept Map</span><br>'
               + p(f'{bdg("Decision trees","purple")} for test selection<br>When to use what'), "purple"),
        )
    )

    mode = st.radio(
        "Section",
        ["🃏 Concept Cards", "📖 Glossary", "📐 Formula Sheet", "🗺 Decision Guide"],
        horizontal=True, key="vocab_mode"
    )

    if "Concept Cards" in mode:
        _concept_cards_section()
    elif "Glossary" in mode:
        _glossary_section()
    elif "Formula Sheet" in mode:
        _formula_section()
    else:
        _decision_guide_section()


# ══════════════════════════════════════════════════════════════════
# CONCEPT CARDS
# ══════════════════════════════════════════════════════════════════

def _concept_cards_section():
    theme = st.selectbox(
        "Theme",
        list(CONCEPT_CARDS.keys()),
        key="concept_theme"
    )
    cards = CONCEPT_CARDS[theme]

    # Render cards in columns of 2 or 3
    if len(cards) == 1:
        st.html(concept_card(**cards[0]))
    elif len(cards) == 2:
        cols = st.columns(2)
        for col, card in zip(cols, cards):
            col.html(concept_card(**card))
    elif len(cards) == 3:
        cols = st.columns(3)
        for col, card in zip(cols, cards):
            col.html(concept_card(**card))
    else:
        # 4 cards: 2x2 grid
        row1, row2 = cards[:2], cards[2:]
        cols = st.columns(2)
        for col, card in zip(cols, row1):
            col.html(concept_card(**card))
        cols2 = st.columns(2)
        for col, card in zip(cols2, row2):
            col.html(concept_card(**card))

    # Bottom: related formula box
    theme_fmls = {
        "Core OLS": ("OLS Quick Reference",
                     "β̂₁ = Cov(X,Y)/Var(X)    β̂₀ = Ȳ−β̂₁X̄\n"
                     "SST = SSR + SSE           R² = SSR/SST\n"
                     "MSE = SSE/(n−k−1)         F = MSR/MSE"),
        "Hypothesis Testing": ("Inference Quick Reference",
                               "t = β̂_j / SE(β̂_j) ~ t(n−k−1)\n"
                               "F = (R²/k) / ((1−R²)/(n−k−1))\n"
                               "95% CI: β̂ ± t_crit × SE(β̂)"),
        "CLRM Assumptions": ("Diagnostic Test Summary",
                              "Normality:      JB ~ χ²(2)  |  SW test\n"
                              "Heterosced.:    BP: LM=n·R²_aux ~ χ²(k)\n"
                              "Autocorr.:      DW = Σ(eₜ−eₜ₋₁)²/Σeₜ²\n"
                              "Multicollin.:   VIF = 1/(1−R²_j)"),
        "Finance Models": ("Finance Regression Summary",
                           "CAPM SCL:   Rᵢ−Rf = α + β(Rₘ−Rf) + ε\n"
                           "FF 3-Factor: Rᵢ−Rf = α + β₁MKT+β₂SMB+β₃HML+ε\n"
                           "Bond yield:  YTM = β₀ + β₁·Duration + ε\n"
                           "P/E model:   P/E = β₀ + β₁ROE + β₂g − β₃Beta + ε"),
        "Model Selection": ("Model Selection Criteria",
                            "Adj R²: 1−(1−R²)(n−1)/(n−k−1)\n"
                            "AIC:    2k − 2ln(L)      [lower = better]\n"
                            "BIC:    k·ln(n) − 2ln(L) [heavier k penalty]"),
    }
    if theme in theme_fmls:
        title, fml_text = theme_fmls[theme]
        render_ib(
            f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">📐 {title}</span>'
            + fml(fml_text),
            "gold"
        )


# ══════════════════════════════════════════════════════════════════
# GLOSSARY
# ══════════════════════════════════════════════════════════════════

def _glossary_section():
    col1, col2 = st.columns([2, 1])
    search = col1.text_input("🔍 Search terms", placeholder="e.g. beta, VIF, heteroscedasticity...", key="gloss_search")
    topic_g = col2.selectbox("Filter by topic", ["All","OLS","CAPM","Diagnostics","Model Fit","Inference"], key="gloss_topic")

    TOPIC_MAP = {
        "OLS":         ["OLS","BLUE"],
        "CAPM":        ["Beta","Jensen","CAPM"],
        "Diagnostics": ["VIF","Durbin","Heteroscedasticity","Jarque"],
        "Model Fit":   ["R²","Adjusted","AIC"],
        "Inference":   ["p-value","Type","BLUE"],
    }

    filtered = GLOSSARY_TERMS
    if search.strip():
        s = search.lower()
        filtered = [t for t in filtered
                    if s in t["term"].lower() or s in t["definition"].lower()
                    or (t.get("finance_note") and s in t["finance_note"].lower())]
    if topic_g != "All":
        kws = TOPIC_MAP.get(topic_g, [])
        filtered = [t for t in filtered if any(k.lower() in t["term"].lower() for k in kws)]

    if not filtered:
        render_ib(rt2("No terms match your search. Try a broader query."), "red")
        return

    st.html(f'<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;'
            f'font-family:{FB};font-size:.83rem;margin-bottom:10px;'
            f'user-select:none;-webkit-user-select:none">'
            f'Showing {len(filtered)} of {len(GLOSSARY_TERMS)} terms</div>')

    for term in filtered:
        st.html(term_card(**term))


# ══════════════════════════════════════════════════════════════════
# FORMULA SHEET
# ══════════════════════════════════════════════════════════════════

def _formula_section():
    # 2×2 grid of formula cards
    sections = list(FORMULA_SECTIONS.items())

    cols1 = st.columns(2)
    for col, (sec_title, rows) in zip(cols1, sections[:2]):
        rows_html = "".join(_row(k, v) for k, v in rows)
        col.html(mini_card(sec_title, "#FFD700",
                           f'<div style="font-family:{FM};font-size:.82rem">{rows_html}</div>'))

    cols2 = st.columns(2)
    for col, (sec_title, rows) in zip(cols2, sections[2:]):
        rows_html = "".join(_row(k, v) for k, v in rows)
        col.html(mini_card(sec_title, "#ADD8E6",
                           f'<div style="font-family:{FM};font-size:.82rem">{rows_html}</div>'))

    # Critical values quick reference
    section_heading("📊 Critical Values at α = 5% (Two-Tailed)")
    st.html(
        table_html(
            ["Test","df / Distribution","Critical Value","Decision Rule"],
            [
                [bdg("t-test","blue"),        txt_s("t(30)"), _f("±2.042"), txt_s("Reject H₀ if |t| > 2.042")],
                [bdg("t-test","blue"),        txt_s("t(60)"), _f("±2.000"), txt_s("Reject H₀ if |t| > 2.000")],
                [bdg("t-test","blue"),        txt_s("t(∞)"),  _f("±1.960"), txt_s("Reject H₀ if |t| > 1.96")],
                [bdg("F-test k=3","gold"),    txt_s("F(3,60)"),_f("2.76"),  txt_s("Reject H₀ if F > 2.76")],
                [bdg("F-test k=5","gold"),    txt_s("F(5,60)"),_f("2.37"),  txt_s("Reject H₀ if F > 2.37")],
                [bdg("JB test","purple"),     txt_s("χ²(2)"),  _f("5.991"), txt_s("Reject normality if JB > 5.991")],
                [bdg("BP test k=3","red"),    txt_s("χ²(3)"),  _f("7.815"), txt_s("Reject homoscedasticity if LM > 7.815")],
                [bdg("Durbin-Watson","orange"),txt_s("n=60"),   _f("1.5 – 2.5"), txt_s("Outside range → autocorrelation")],
            ]
        )
    )

    # Degrees of freedom guide
    section_heading("📐 Degrees of Freedom Guide")
    st.html(
        table_html(
            ["Test","df Numerator","df Denominator","Notes"],
            [
                [bdg("t-test (SLR β̂₁)","blue"),   txt_s("—"),   txt_s("n − 2"),    txt_s("One slope estimated + intercept")],
                [bdg("t-test (MLR β̂_j)","blue"),   txt_s("—"),   txt_s("n − k − 1"), txt_s("k slopes + intercept")],
                [bdg("F-test (MLR)","gold"),         txt_s("k"),   txt_s("n − k − 1"), txt_s("k = number of predictors")],
                [bdg("Jarque-Bera","purple"),         txt_s("2"),   txt_s("— (χ²)"),  txt_s("Skewness + excess kurtosis terms")],
                [bdg("Breusch-Pagan","red"),          txt_s("k"),   txt_s("— (χ²)"),  txt_s("k = predictors in original model")],
            ]
        )
    )


# ══════════════════════════════════════════════════════════════════
# DECISION GUIDE
# ══════════════════════════════════════════════════════════════════

def _decision_guide_section():
    render_card("🗺 Decision Guide — When to Use What",
        p(f'Use these decision trees to choose the right test, model, or remedy for your regression problem.')
    )

    # Which regression?
    section_heading("1️⃣  Which Regression Model?")
    st.html(
        table_html(
            ["Situation","Model","Key Feature"],
            [
                [txt_s("One Y, one X"),                    bdg("SLR","blue"),    txt_s("CAPM beta estimation")],
                [txt_s("One Y, multiple X"),               bdg("MLR","gold"),    txt_s("Fama-French, credit risk")],
                [txt_s("Y is binary (default / no default)"),bdg("Logit/Probit","red"), txt_s("PD modelling")],
                [txt_s("Heteroscedasticity confirmed"),     bdg("WLS / GLS","orange"), txt_s("Bond portfolio weighting")],
                [txt_s("Multicollinearity is severe"),      bdg("Ridge / LASSO","purple"), txt_s("Factor-saturated models")],
                [txt_s("Time series with autocorrelation"), bdg("GLS / ARIMA","green"), txt_s("Macro factor models")],
            ]
        )
    )

    # Which diagnostic test?
    section_heading("2️⃣  Which Diagnostic Test?")
    st.html(
        table_html(
            ["You suspect...","Test to run","Statistic","Remedy if rejected"],
            [
                [txt_s("Non-normal residuals"),         bdg("Jarque-Bera","gold"),     _f("JB ~ χ²(2)"),           txt_s("Robust SE, bootstrap")],
                [txt_s("Non-normal (small n < 50)"),    bdg("Shapiro-Wilk","blue"),    _f("W ~ [0,1]"),             txt_s("Transform Y, non-parametric")],
                [txt_s("Heteroscedasticity"),            bdg("Breusch-Pagan","red"),    _f("LM=n·R²~χ²(k)"),        txt_s("HC3 robust SE, WLS")],
                [txt_s("Autocorrelation (AR1)"),         bdg("Durbin-Watson","orange"), _f("DW = Σ(Δe)²/Σe²"),      txt_s("Newey-West HAC, GLS")],
                [txt_s("Higher-order autocorr."),        bdg("Breusch-Godfrey","blue"), _f("LM ~ χ²(p)"),           txt_s("HAC SE, AR(p) model")],
                [txt_s("Multicollinearity"),              bdg("VIF","purple"),           _f("1/(1−R²_j)"),            txt_s("Ridge, PCA, drop variable")],
                [txt_s("Functional misspecification"),   bdg("RESET","green"),          _f("F ~ F(2, n−k−3)"),       txt_s("Add polynomial or log terms")],
                [txt_s("Structural break"),               bdg("Chow test","orange"),     _f("F ~ F(k+1, n−2k−2)"),   txt_s("Include dummy or split sample")],
            ]
        )
    )

    # Which model selection criterion?
    section_heading("3️⃣  Which Model Selection Criterion?")
    two_left = (
        ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Comparing models with same n:</span><br>'
           + steps_html([
               ("Use Adjusted R²",  "Increases only if new variable genuinely improves fit. Compare same dependent variable."),
               ("Use AIC",          "When the goal is prediction accuracy. Penalises 2 per parameter. Chooses larger models than BIC."),
               ("Use BIC",          "When the goal is identifying the true model structure. Heavier penalty: k·ln(n). Preferred for inference."),
               ("Use RMSE",         "For comparing forecast accuracy on held-out test data. Same units as Y."),
           ]), "gold")
    )
    two_right = (
        ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">Quick rules:</span><br>'
           + p(f'{hl("Both AIC and BIC agree")} → clear winner.<br>'
               f'{hl("AIC selects larger model")} → prefer for prediction tasks.<br>'
               f'{hl("BIC selects smaller model")} → prefer for explanation.<br><br>'
               f'{bdg("Never","red")} compare R² across models with different k.<br>'
               f'{bdg("Always","green")} use Adj R² or information criteria for model selection.'),
           "blue")
    )
    st.html(two_col(two_left, two_right))

    # VIF action guide
    section_heading("4️⃣  Multicollinearity Action Guide")
    st.html(
        table_html(
            ["VIF Value","Severity","Action"],
            [
                [_f("VIF < 5"),    bdg("OK","green"),          txt_s("No action needed. Report and proceed.")],
                [_f("5 ≤ VIF < 10"),bdg("Moderate","orange"),  txt_s("Investigate. Monitor coefficient stability across subsamples.")],
                [_f("VIF ≥ 10"),   bdg("Serious","red"),        txt_s("Take action: Ridge regression, drop variable, PCA, or collect more data.")],
                [_f("VIF = ∞"),    bdg("Perfect collinearity","red"), txt_s("One X is exact linear combination of others — model is unidentified.")],
            ]
        )
    )

    # Assumption violation impact table
    section_heading("5️⃣  CLRM Violation Impact Summary")
    st.html(
        table_html(
            ["Violation","β̂ Biased?","SE Biased?","t/F Valid?","Remedy"],
            [
                [bdg("Heteroscedasticity","red"),    gt("No"),  rt2("Yes"), rt2("No"),  txt_s("HC3 robust SE, WLS")],
                [bdg("Autocorrelation","orange"),     gt("No"),  rt2("Yes"), rt2("No"),  txt_s("HAC Newey-West, GLS")],
                [bdg("Multicollinearity","purple"),   gt("No"),  rt2("Yes"), org("Weak"), txt_s("Ridge, PCA, drop/combine")],
                [bdg("Non-normality (large n)","gold"),gt("No"), gt("No"),  gt("Yes"), txt_s("CLT applies; no action in large n")],
                [bdg("Non-normality (small n)","gold"),gt("No"),gt("No"),  rt2("No"),  txt_s("Bootstrap CI, transform Y")],
                [bdg("Omitted variable","blue"),      rt2("Yes"),rt2("Yes"), rt2("No"), txt_s("Add the variable; IV if endogenous")],
                [bdg("Endogeneity","red"),             rt2("Yes"),rt2("Yes"), rt2("No"), txt_s("Instrumental variables (IV/2SLS)")],
            ]
        )
    )
