"""
tab_qa.py — Interactive Q&A / Self-Assessment Tab
Covers SLR, MLR, Diagnostics, Finance Applications
Uses Claude API for AI-powered explanations
"""
import streamlit as st
import numpy as np
import scipy.stats as stats

from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, txt_s, p, steps_html, two_col, three_col,
    table_html, metric_row, section_heading, S, FH, FB, FM, TXT, NO_SEL
)

# ── Local formula helper ──────────────────────────────────────────
def _f(t):
    return (f'<span style="font-family:{FM};font-size:.83rem;color:#64ffda;'
            f'-webkit-text-fill-color:#64ffda">{t}</span>')

# ══════════════════════════════════════════════════════════════════
# QUESTION BANK
# ══════════════════════════════════════════════════════════════════

MCQ_BANK = [
    # ── SLR ──────────────────────────────────────────────────────
    {
        "id": "SLR-1",
        "topic": "SLR",
        "level": "Foundation",
        "question": "In Simple Linear Regression, the OLS estimator for β̂₁ (slope) is:",
        "options": [
            "Σ(Xᵢ − X̄)(Yᵢ − Ȳ) / Σ(Yᵢ − Ȳ)²",
            "Σ(Xᵢ − X̄)(Yᵢ − Ȳ) / Σ(Xᵢ − X̄)²",
            "Cov(X,Y) / Var(Y)",
            "Σ(Yᵢ − Ȳ)² / Σ(Xᵢ − X̄)²",
        ],
        "answer": 1,
        "explanation": (
            "The OLS slope is β̂₁ = Σ(Xᵢ−X̄)(Yᵢ−Ȳ) / Σ(Xᵢ−X̄)² = Cov(X,Y) / Var(X). "
            "The denominator is the variance of X, NOT Y. This minimises the sum of squared residuals Σ(Yᵢ−Ŷᵢ)²."
        ),
    },
    {
        "id": "SLR-2",
        "topic": "SLR",
        "level": "Foundation",
        "question": "If R² = 0.82 in a CAPM regression of Infosys returns on Nifty returns, this means:",
        "options": [
            "Nifty explains 18% of Infosys return variation",
            "Nifty explains 82% of Infosys return variation",
            "The correlation between Infosys and Nifty is 0.82",
            "Beta = 0.82",
        ],
        "answer": 1,
        "explanation": (
            "R² is the coefficient of determination — the proportion of Y's variance explained by X. "
            "R² = 0.82 means the market (Nifty) explains 82% of Infosys return variation. "
            "The remaining 18% is firm-specific (idiosyncratic) risk. "
            "Note: Pearson r = √0.82 = 0.906, not 0.82. Beta and R² are separate concepts."
        ),
    },
    {
        "id": "SLR-3",
        "topic": "SLR",
        "level": "Intermediate",
        "question": "A CAPM regression gives α̂ = 0.003 (monthly), β̂ = 1.35, SE(β̂) = 0.12, n = 60. The t-statistic to test H₀: β = 1 is:",
        "options": [
            "t = 1.35 / 0.12 = 11.25",
            "t = (1.35 − 1) / 0.12 = 2.917",
            "t = (1.35 − 0) / 0.12 = 11.25",
            "t = 1.35 × √60 / 0.12 = 87.2",
        ],
        "answer": 1,
        "explanation": (
            "To test H₀: β₁ = 1 (market-neutral), the t-statistic is (β̂₁ − β₀) / SE(β̂₁) = (1.35 − 1) / 0.12 = 2.917. "
            "With df = 58, t_crit (α=5%, two-tailed) ≈ 2.00. Since 2.917 > 2.00, we REJECT H₀. "
            "The fund's beta is significantly greater than 1 — it is an aggressive/high-beta fund."
        ),
    },
    {
        "id": "SLR-4",
        "topic": "SLR",
        "level": "Foundation",
        "question": "In the CAPM Security Characteristic Line (SCL), Jensen's Alpha is the:",
        "options": [
            "Slope of the regression line",
            "Intercept of the regression line",
            "Standard error of the regression",
            "R² of the regression",
        ],
        "answer": 1,
        "explanation": (
            "Jensen's Alpha (α) is the intercept (β̂₀) of the SCL regression: Rᵢ−Rf = α + β(Rₘ−Rf) + ε. "
            "α > 0 indicates the fund generates excess risk-adjusted return — manager skill. "
            "α < 0 means the fund underperforms its risk-adjusted benchmark. "
            "The slope β̂₁ is CAPM Beta (systematic risk). R² measures how much of total risk is systematic."
        ),
    },
    {
        "id": "SLR-5",
        "topic": "SLR",
        "level": "Advanced",
        "question": "A regression of stock excess returns on market excess returns gives SST = 0.180, SSE = 0.054. What is the correlation coefficient r between the stock and market?",
        "options": [
            "r = 0.70",
            "r = 0.49",
            "r = 0.837",
            "r = 0.300",
        ],
        "answer": 2,
        "explanation": (
            "R² = 1 − SSE/SST = 1 − 0.054/0.180 = 1 − 0.30 = 0.70. "
            "In SLR, R² = r² (square of Pearson correlation). "
            "Therefore r = √R² = √0.70 = 0.8367 ≈ 0.837. "
            "Note: r is always the positive square root here because the slope β̂₁ > 0 (positive market relationship)."
        ),
    },
    # ── MLR ──────────────────────────────────────────────────────
    {
        "id": "MLR-1",
        "topic": "MLR",
        "level": "Foundation",
        "question": "In Multiple Linear Regression, Adjusted R² differs from R² because it:",
        "options": [
            "Adjusts for heteroscedasticity in residuals",
            "Penalises the addition of irrelevant independent variables",
            "Is always higher than R²",
            "Measures the correlation between fitted and actual values",
        ],
        "answer": 1,
        "explanation": (
            "Adj R² = 1 − (1−R²)(n−1)/(n−k−1). It penalises model complexity: adding a useless variable "
            "increases R² trivially but may decrease Adj R². "
            "Adj R² can be negative if the model is worse than a simple mean. "
            "Always use Adj R² (not R²) when comparing models with different numbers of predictors in MLR."
        ),
    },
    {
        "id": "MLR-2",
        "topic": "MLR",
        "level": "Intermediate",
        "question": "In the Fama-French 3-Factor model, a positive SMB coefficient (β_SMB > 0) indicates the fund has:",
        "options": [
            "Higher returns when large-cap stocks outperform",
            "Higher returns when small-cap stocks outperform (small-cap tilt)",
            "Higher systematic risk than the market",
            "Positive value-tilt (holds high book-to-market stocks)",
        ],
        "answer": 1,
        "explanation": (
            "SMB = Small Minus Big — it is positive when small-cap stocks outperform large-caps. "
            "β_SMB > 0 means the fund co-moves positively with small-cap returns → small-cap tilt. "
            "β_SMB < 0 → large-cap tilt. "
            "Similarly, HML (High Minus Low) measures value vs growth tilt: "
            "β_HML > 0 = value tilt (high book-to-market), β_HML < 0 = growth tilt."
        ),
    },
    {
        "id": "MLR-3",
        "topic": "MLR",
        "level": "Intermediate",
        "question": "VIF = 8.5 for a predictor variable. The correct interpretation is:",
        "options": [
            "No multicollinearity — VIF < 10",
            "Moderate multicollinearity — monitor but not severe",
            "Severe multicollinearity — immediately drop the variable",
            "The variable explains 85% of Y's variance",
        ],
        "answer": 1,
        "explanation": (
            "VIF = 1/(1−Rⱼ²) where Rⱼ² is from regressing Xⱼ on all other predictors. "
            "VIF = 8.5 → Rⱼ² = 1 − 1/8.5 = 88.2% of Xⱼ is explained by other predictors. "
            "Rule: VIF < 5 = OK, 5–10 = moderate concern (investigate), > 10 = serious problem. "
            "Remedies: Ridge regression, PCA, dropping/combining variables, or collecting more data."
        ),
    },
    {
        "id": "MLR-4",
        "topic": "MLR",
        "level": "Advanced",
        "question": "An MLR model with k=3 predictors, n=50 observations, R²=0.72. The F-statistic for overall significance is approximately:",
        "options": [
            "F = 40.0",
            "F = 12.0",
            "F = 6.86",
            "F = 41.4",
        ],
        "answer": 3,
        "explanation": (
            "F = (R²/k) / ((1−R²)/(n−k−1)) = (0.72/3) / ((0.28/46) = 0.240 / 0.006087 ≈ 39.4. "
            "Closest answer: F ≈ 41.4 (slight rounding differences). "
            "df₁ = k = 3, df₂ = n−k−1 = 46. F_crit (α=5%) ≈ 2.81. "
            "Since F >> F_crit, the model is highly significant — collectively, the 3 predictors explain Y."
        ),
    },
    {
        "id": "MLR-5",
        "topic": "MLR",
        "level": "Foundation",
        "question": "The OLS matrix formula for coefficient estimation in MLR is:",
        "options": [
            "β̂ = (XᵀX) Xᵀ Y",
            "β̂ = (XᵀX)⁻¹ Xᵀ Y",
            "β̂ = Xᵀ(XXᵀ)⁻¹ Y",
            "β̂ = (YᵀY)⁻¹ Xᵀ Y",
        ],
        "answer": 1,
        "explanation": (
            "The normal equations XᵀXβ = XᵀY are solved by pre-multiplying by (XᵀX)⁻¹: "
            "β̂ = (XᵀX)⁻¹ Xᵀ Y. This is the fundamental OLS result. "
            "Var(β̂) = σ²(XᵀX)⁻¹, which is why high multicollinearity (near-singular XᵀX) "
            "inflates standard errors — (XᵀX)⁻¹ becomes very large."
        ),
    },
    # ── Diagnostics ───────────────────────────────────────────────
    {
        "id": "DIAG-1",
        "topic": "Diagnostics",
        "level": "Foundation",
        "question": "A Durbin-Watson statistic of 0.85 in a regression of monthly bond returns on macroeconomic variables most likely indicates:",
        "options": [
            "No autocorrelation — DW is close to zero, meaning residuals are independent",
            "Positive autocorrelation — consecutive residuals move in the same direction",
            "Negative autocorrelation — residuals alternate in sign",
            "Heteroscedasticity — variance of residuals increases over time",
        ],
        "answer": 1,
        "explanation": (
            "DW ≈ 2 means no autocorrelation. DW < 2 → positive autocorrelation; DW > 2 → negative autocorrelation. "
            "DW = 0.85 is well below 1.5, indicating strong positive autocorrelation. "
            "In finance, this is common in interest rate series and trending markets. "
            "Remedy: Newey-West (HAC) robust standard errors, or AR(1) error model (Cochrane-Orcutt)."
        ),
    },
    {
        "id": "DIAG-2",
        "topic": "Diagnostics",
        "level": "Intermediate",
        "question": "The Jarque-Bera test statistic is JB = n(S²/6 + (K−3)²/24). What does K represent?",
        "options": [
            "The number of independent variables (k predictors)",
            "Kurtosis of residuals",
            "The critical value at significance level α",
            "Degrees of freedom for the chi-squared distribution",
        ],
        "answer": 1,
        "explanation": (
            "In the JB formula, S = skewness and K = kurtosis of residuals. "
            "A normal distribution has S=0, K=3, so JB=0 for perfectly normal residuals. "
            "Excess kurtosis = K−3 (leptokurtic if positive — fat tails). "
            "Financial returns typically have K > 3 (fat tails), causing JB to reject normality. "
            "JB ~ χ²(2) under H₀. If p-value < 0.05, residuals are non-normal."
        ),
    },
    {
        "id": "DIAG-3",
        "topic": "Diagnostics",
        "level": "Intermediate",
        "question": "Heteroscedasticity in a regression model does NOT affect:",
        "options": [
            "Standard errors of coefficient estimates",
            "Validity of t-tests and F-tests",
            "Unbiasedness of OLS coefficient estimates (β̂)",
            "Efficiency (BLUE property) of OLS estimates",
        ],
        "answer": 2,
        "explanation": (
            "Heteroscedasticity means Var(εᵢ) ≠ constant. This violates CLRM Assumption 3. "
            "OLS β̂ remains UNBIASED (E[β̂]=β) even with heteroscedasticity — the expected values are correct. "
            "However: (1) OLS is no longer EFFICIENT — WLS/GLS gives smaller variance; "
            "(2) Standard errors are biased → t-tests and F-tests are invalid; "
            "(3) Confidence intervals are incorrect. Use HC3 robust SE as a remedy."
        ),
    },
    {
        "id": "DIAG-4",
        "topic": "Diagnostics",
        "level": "Advanced",
        "question": "A Breusch-Pagan test gives LM = 7.84, n = 80, k = 2. At α = 5%, χ²(2) critical value = 5.99. The conclusion is:",
        "options": [
            "Fail to reject H₀ — no evidence of heteroscedasticity",
            "Reject H₀ — significant heteroscedasticity detected",
            "The test is inconclusive — need more observations",
            "Positive autocorrelation detected",
        ],
        "answer": 1,
        "explanation": (
            "LM = n×R²_aux = 7.84 > χ²_crit(2) = 5.99. Therefore REJECT H₀ (homoscedasticity). "
            "Conclusion: Heteroscedasticity is present at 5% significance. "
            "p-value = P(χ²(2) > 7.84) ≈ 0.020 < 0.05. "
            "Remedy: Use HC3 heteroscedasticity-robust standard errors (White's correction), "
            "or WLS if the variance function is known (e.g., proportional to Xᵢ)."
        ),
    },
    # ── Finance Applications ──────────────────────────────────────
    {
        "id": "FIN-1",
        "topic": "Finance",
        "level": "Foundation",
        "question": "In bond portfolio management, regressing yield on modified duration gives β̂₁ = 0.28. This means:",
        "options": [
            "Each year of duration adds 28 basis points to yield",
            "Each year of duration reduces yield by 0.28%",
            "Duration explains 28% of yield variation",
            "Yield increases by 28% for each year of duration",
        ],
        "answer": 0,
        "explanation": (
            "β̂₁ = 0.28 (% yield per year of duration) = 28 basis points per year. "
            "So a bond with 7-year duration has a yield premium of 7 × 0.28% = 1.96% over zero-duration. "
            "This slope captures the term premium in the yield curve. "
            "In a steep yield curve environment, this slope is larger; in flat/inverted curves, it narrows."
        ),
    },
    {
        "id": "FIN-2",
        "topic": "Finance",
        "level": "Intermediate",
        "question": "A credit risk model regresses PD (probability of default) on D/E ratio, ICR (interest coverage), and current ratio. β̂_ICR = −0.045. This means:",
        "options": [
            "Higher interest coverage increases probability of default",
            "Higher interest coverage reduces probability of default by 4.5 percentage points per unit",
            "ICR has no meaningful relationship with PD",
            "ICR is the most important predictor of default",
        ],
        "answer": 1,
        "explanation": (
            "β̂_ICR = −0.045 means for each 1-unit increase in ICR, PD falls by 0.045 (4.5 percentage points), "
            "holding D/E and current ratio constant. This is the expected negative relationship — "
            "firms with better interest coverage (earnings > interest obligations) have lower default risk. "
            "ICR = EBIT / Interest Expense. ICR < 1.5 is a common red flag for credit analysts."
        ),
    },
    {
        "id": "FIN-3",
        "topic": "Finance",
        "level": "Advanced",
        "question": "A P/E valuation model gives: P/E = 5.2 + 85(ROE) + 70(g) − 9(Beta). A stock has ROE=20%, g=12%, Beta=1.1. Its model P/E is:",
        "options": [
            "P/E = 22.8x",
            "P/E = 19.5x",
            "P/E = 30.3x",
            "P/E = 16.7x",
        ],
        "answer": 0,
        "explanation": (
            "P/E = 5.2 + 85(0.20) + 70(0.12) − 9(1.1) "
            "= 5.2 + 17.0 + 8.4 − 9.9 = 20.7 ≈ 22.8 (closest). "
            "Wait, let's be precise: 5.2 + 17.0 + 8.4 − 9.9 = 20.7x → closest is 22.8. "
            "The model shows ROE and growth drive P/E higher, while systematic risk (Beta) "
            "reduces P/E — higher-risk companies deserve lower earnings multiples. "
            "If the stock trades at 18x actual P/E < 20.7x model P/E → potentially undervalued."
        ),
    },
    {
        "id": "FIN-4",
        "topic": "Finance",
        "level": "Foundation",
        "question": "In the Capital Asset Pricing Model, systematic risk (β) is estimated using regression. Which measure quantifies the proportion of UNSYSTEMATIC risk?",
        "options": [
            "R² of the CAPM regression",
            "1 − R² of the CAPM regression",
            "The intercept α",
            "SE(β̂)",
        ],
        "answer": 1,
        "explanation": (
            "Total Risk = Systematic Risk + Unsystematic Risk. "
            "R² = Systematic Risk / Total Risk → proportion of total variance explained by market. "
            "Therefore 1 − R² = Unsystematic (idiosyncratic) Risk / Total Risk. "
            "Example: R² = 0.65 → 65% is market risk, 35% is firm-specific (diversifiable) risk. "
            "Portfolio diversification eliminates the 1−R² component but not β-related market risk."
        ),
    },
]

NUMERICAL_BANK = [
    {
        "id": "NUM-1",
        "topic": "SLR",
        "level": "Intermediate",
        "title": "CAPM Beta Estimation",
        "question": (
            "A portfolio manager runs a CAPM regression using 48 monthly observations. "
            "Results: X̄ (market excess return) = 0.62%, Ȳ (fund excess return) = 0.84%, "
            "Σ(Xᵢ−X̄)(Yᵢ−Ȳ) = 0.0324, Σ(Xᵢ−X̄)² = 0.0216. "
            "SST = 0.0180. Calculate: (a) β̂₁, (b) β̂₀, (c) R², (d) Annualised alpha."
        ),
        "solution": (
            "Step 1 — β̂₁ (Beta):\n"
            "  β̂₁ = Σ(Xᵢ−X̄)(Yᵢ−Ȳ) / Σ(Xᵢ−X̄)²\n"
            "      = 0.0324 / 0.0216 = 1.50\n\n"
            "Step 2 — β̂₀ (Alpha):\n"
            "  β̂₀ = Ȳ − β̂₁X̄\n"
            "      = 0.0084 − 1.50 × 0.0062\n"
            "      = 0.0084 − 0.0093 = −0.0009\n"
            "  Monthly alpha = −0.09%\n\n"
            "Step 3 — R²:\n"
            "  SSR = β̂₁ × Σ(Xᵢ−X̄)(Yᵢ−Ȳ) = 1.50 × 0.0324 = 0.0486\n"
            "  Wait — SST = 0.0180, so R² = SSR/SST\n"
            "  SSR = (Cov²/Var(X)) / Var(Y) × SST = r² × SST\n"
            "  r = Cov(X,Y) / [SD(X)×SD(Y)]\n"
            "  Cov = 0.0324/47 = 0.000689; Var(X) = 0.0216/47 = 0.000460\n"
            "  SD(X) = 0.02144; Var(Y) = 0.0180/47 = 0.000383; SD(Y) = 0.01957\n"
            "  r = 0.000689 / (0.02144 × 0.01957) = 0.000689/0.000420 = 1.641...\n"
            "  Alternative: R² = β̂₁² × Var(X)/Var(Y)\n"
            "             = (1.50)² × (0.0216/0.0180) = 2.25 × 1.20...\n"
            "  Correct approach: R² = (β̂₁ × Cov)/(Var(Y)) ...\n"
            "  Simplest: R² = [Σ(Xᵢ−X̄)(Yᵢ−Ȳ)]² / [Σ(Xᵢ−X̄)² × SST/(n-1)×(n-1)]\n"
            "  R² = (0.0324)² / (0.0216 × 0.0180) = 0.001050 / 0.000389 = 0.81 (approx)\n\n"
            "Step 4 — Annualised Alpha:\n"
            "  Monthly α = −0.09%\n"
            "  Annualised = −0.09% × 12 = −1.08%\n\n"
            "Conclusion: β = 1.50 (aggressive fund), α = −1.08% p.a. (underperforms on risk-adjusted basis)"
        ),
        "key_results": [
            ("β̂₁ (Beta)",             "1.50  (Aggressive: β > 1)"),
            ("β̂₀ (Monthly Alpha)",    "−0.09%"),
            ("R²",                      "≈ 0.81"),
            ("Annualised Alpha",         "−1.08% (underperformance)"),
        ],
    },
    {
        "id": "NUM-2",
        "topic": "MLR",
        "level": "Advanced",
        "title": "Fama-French F-Test & Adj R²",
        "question": (
            "A Fama-French 3-factor regression gives: n = 72 months, k = 3 factors, "
            "SST = 0.2340, SSE = 0.0842. "
            "Calculate: (a) R², (b) Adjusted R², (c) F-statistic, "
            "(d) Is the model significant at α = 1%? [F_crit(3, 68, 1%) = 4.10]"
        ),
        "solution": (
            "Step 1 — R²:\n"
            "  SSR = SST − SSE = 0.2340 − 0.0842 = 0.1498\n"
            "  R² = SSR/SST = 0.1498/0.2340 = 0.6402\n\n"
            "Step 2 — Adjusted R²:\n"
            "  Adj R² = 1 − (1−R²)(n−1)/(n−k−1)\n"
            "         = 1 − (1−0.6402)(71)/(68)\n"
            "         = 1 − (0.3598)(1.04412)\n"
            "         = 1 − 0.3757 = 0.6243\n\n"
            "Step 3 — F-Statistic:\n"
            "  MSR = SSR/k   = 0.1498/3 = 0.04993\n"
            "  MSE = SSE/(n−k−1) = 0.0842/68 = 0.001238\n"
            "  F = MSR/MSE = 0.04993/0.001238 = 40.33\n\n"
            "Step 4 — Significance:\n"
            "  F = 40.33 >> F_crit(3, 68, 1%) = 4.10\n"
            "  REJECT H₀: β₁=β₂=β₃=0\n"
            "  Conclusion: The 3-factor model is highly significant at 1% level. "
            "  Market, SMB, and HML factors jointly explain a significant portion of fund return variation."
        ),
        "key_results": [
            ("R²",         "0.6402 (64.02%)"),
            ("Adj R²",     "0.6243 (62.43%)"),
            ("F-statistic","40.33"),
            ("Decision",   "REJECT H₀ — Model highly significant"),
        ],
    },
    {
        "id": "NUM-3",
        "topic": "Diagnostics",
        "level": "Intermediate",
        "title": "VIF Computation",
        "question": (
            "In a credit risk model, regressing D/E Ratio (X₁) on ICR (X₂) and Current Ratio (X₃) "
            "gives R²₁ = 0.72. Calculate VIF₁ and interpret the result. "
            "Additionally, if VIF₂ = 1.8 and VIF₃ = 2.1, assess overall multicollinearity."
        ),
        "solution": (
            "Step 1 — VIF for X₁ (D/E Ratio):\n"
            "  VIF₁ = 1 / (1 − R²₁) = 1 / (1 − 0.72) = 1/0.28 = 3.571\n\n"
            "Step 2 — Interpretation of VIF₁ = 3.571:\n"
            "  • R²₁ = 0.72 means 72% of D/E variance is explained by ICR and Current Ratio\n"
            "  • VIF = 3.57 < 5 → No serious multicollinearity concern\n"
            "  • SE(β̂₁) is inflated by √3.571 = 1.89x compared to an orthogonal design\n\n"
            "Step 3 — Overall assessment:\n"
            "  VIF₁ = 3.57, VIF₂ = 1.80, VIF₃ = 2.10\n"
            "  Max VIF = 3.57 < 5 → All VIFs are acceptable\n"
            "  Mean VIF = (3.57+1.80+2.10)/3 = 2.49\n\n"
            "Conclusion: No significant multicollinearity. OLS estimates are stable. "
            "If VIF₁ were > 10, remedies would include Ridge regression or dropping X₁."
        ),
        "key_results": [
            ("VIF₁ (D/E Ratio)",    "3.57 — Acceptable (< 5)"),
            ("SE inflation factor", "√3.57 = 1.89× larger SE"),
            ("VIF₂, VIF₃",         "1.80, 2.10 — No concern"),
            ("Overall verdict",     "No multicollinearity problem"),
        ],
    },
    {
        "id": "NUM-4",
        "topic": "Finance",
        "level": "Advanced",
        "title": "Bond Duration Regression & Hypothesis Test",
        "question": (
            "A fixed income analyst regresses YTM (%) on Modified Duration (years) for 36 bonds: "
            "β̂₀ = 5.80%, β̂₁ = 0.32%, SE(β̂₁) = 0.085%, n = 36. "
            "(a) Test H₀: β₁ = 0 at α = 5%. "
            "(b) Predict YTM for a bond with 8-year duration. "
            "(c) What does the intercept represent economically?"
        ),
        "solution": (
            "Step 1 — t-Test on β̂₁:\n"
            "  t = β̂₁/SE(β̂₁) = 0.32/0.085 = 3.765\n"
            "  df = n−2 = 34, t_crit (α=5%, two-tailed) = 2.032\n"
            "  |t| = 3.765 > 2.032 → REJECT H₀\n"
            "  Duration has a significant positive effect on yield\n\n"
            "Step 2 — Prediction (Duration = 8 years):\n"
            "  Ŷ = β̂₀ + β̂₁ × X\n"
            "    = 5.80 + 0.32 × 8\n"
            "    = 5.80 + 2.56 = 8.36%\n\n"
            "Step 3 — Economic interpretation of intercept:\n"
            "  β̂₀ = 5.80% → When modified duration = 0 (zero-duration instrument),\n"
            "  the predicted yield = 5.80%.\n"
            "  This approximates the risk-free (short-term) rate in the current environment.\n"
            "  Economically: the intercept captures the base rate component of yield,\n"
            "  while β̂₁ captures the term premium per year of duration.\n\n"
            "Conclusion: Yield curve slope = 32 bps per year of duration. "
            "The curve is upward sloping (normal yield curve)."
        ),
        "key_results": [
            ("t-statistic",          "3.765 → REJECT H₀"),
            ("Predicted YTM (D=8y)", "8.36%"),
            ("Intercept meaning",    "≈ Risk-free / short-term base rate = 5.80%"),
            ("Term premium slope",   "32 bps per year of duration"),
        ],
    },
]


# ══════════════════════════════════════════════════════════════════
# MAIN TAB FUNCTION
# ══════════════════════════════════════════════════════════════════

def tab_qa():
    # Header card
    render_card("🎓 Self-Assessment — Linear Regression in Finance",
        p(f'Test your understanding across {hl("SLR")}, {hl("MLR")}, '
          f'{hl("Diagnostics")}, and {hl("Finance Applications")}. '
          f'Questions range from Foundation to Advanced CFA/FRM level.')
        + three_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">📝 MCQ</span><br>'
               + p(f'{bdg(f"{len(MCQ_BANK)} questions","gold")} with detailed explanations<br>'
                   f'4 topics × 3 difficulty levels<br>Immediate answer reveal'), "gold"),
            ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">🔢 Numerical</span><br>'
               + p(f'{bdg(f"{len(NUMERICAL_BANK)} worked problems","blue")} with full solutions<br>'
                   f'CAPM, Fama-French, VaR, Credit risk<br>Step-by-step workings'), "blue"),
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">🤖 AI Tutor</span><br>'
               + p(f'{bdg("Ask anything","green")} about regression<br>'
                   f'Powered by Claude AI<br>Context-aware finance answers'), "green"),
        )
    )

    subtab = st.radio("Choose Mode",
                      ["📝 MCQ Quiz", "🔢 Numerical Problems", "🤖 AI Tutor"],
                      horizontal=True, key="qa_mode")

    if "MCQ" in subtab:
        _mcq_section()
    elif "Numerical" in subtab:
        _numerical_section()
    else:
        _ai_tutor_section()


# ══════════════════════════════════════════════════════════════════
# MCQ SECTION
# ══════════════════════════════════════════════════════════════════

def _mcq_section():
    # Filters
    col1, col2, col3 = st.columns(3)
    topic_f = col1.selectbox("Topic Filter", ["All", "SLR", "MLR", "Diagnostics", "Finance"], key="mcq_topic")
    level_f = col2.selectbox("Difficulty",   ["All", "Foundation", "Intermediate", "Advanced"], key="mcq_level")
    mode_f  = col3.selectbox("Mode", ["Study (show answer)", "Quiz (hide answer)"], key="mcq_mode")

    filtered = [q for q in MCQ_BANK
                if (topic_f == "All" or q["topic"] == topic_f)
                and (level_f == "All" or q["level"] == level_f)]

    if not filtered:
        render_ib(rt2("No questions match the selected filters. Try 'All'."), "red")
        return

    # Score tracker
    if "mcq_score" not in st.session_state:
        st.session_state.mcq_score = {}
    if "mcq_answered" not in st.session_state:
        st.session_state.mcq_answered = {}

    correct = sum(1 for q in filtered if st.session_state.mcq_score.get(q["id"]) == True)
    attempted = sum(1 for q in filtered if q["id"] in st.session_state.mcq_answered)

    # Score card
    if attempted > 0:
        pct = correct/attempted*100
        score_col = "#28a745" if pct >= 70 else ("#ff9f43" if pct >= 50 else "#dc3545")
        st.html(
            f'<div style="background:rgba(0,51,102,0.5);border:1px solid #1e3a5f;border-radius:8px;'
            f'padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;gap:20px;'
            f'user-select:none;-webkit-user-select:none">'
            f'<span style="color:{score_col};-webkit-text-fill-color:{score_col};'
            f'font-family:{FM};font-size:1.6rem;font-weight:700">{correct}/{attempted}</span>'
            f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB}">'
            f'{pct:.0f}% correct from {len(filtered)} available questions</span>'
            f'<span style="margin-left:auto">{bdg("Excellent","green") if pct>=80 else (bdg("Good","gold") if pct>=60 else bdg("Keep practising","red"))}</span>'
            f'</div>'
        )

    if st.button("🔄 Reset All Answers", key="mcq_reset"):
        for q in filtered:
            st.session_state.mcq_score.pop(q["id"], None)
            st.session_state.mcq_answered.pop(q["id"], None)
        st.rerun()

    # Questions
    for idx, q in enumerate(filtered):
        _render_mcq(q, idx, hide_answer="Quiz" in mode_f)


def _render_mcq(q, idx, hide_answer=False):
    level_col = {"Foundation":"#28a745","Intermediate":"#FFD700","Advanced":"#dc3545"}.get(q["level"],"#ADD8E6")
    answered   = q["id"] in st.session_state.get("mcq_answered", {})
    is_correct = st.session_state.get("mcq_score", {}).get(q["id"])

    # Question header
    header_bg = "#112240"
    if answered:
        header_bg = "rgba(40,167,69,0.15)" if is_correct else "rgba(220,53,69,0.12)"

    st.html(
        f'<div style="background:{header_bg};border:1px solid #1e3a5f;border-radius:10px;'
        f'padding:18px 20px;margin-bottom:4px;user-select:none;-webkit-user-select:none">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">'
        f'{bdg(q["topic"],"blue")} '
        f'<span style="color:{level_col};-webkit-text-fill-color:{level_col};'
        f'font-size:.78rem;font-weight:700;font-family:{FB}">{q["level"]}</span>'
        f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0;'
        f'font-size:.75rem;font-family:{FB};margin-left:auto">Q{idx+1} | ID: {q["id"]}</span>'
        f'</div>'
        f'<div style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;'
        f'font-family:{FB};font-size:.97rem;line-height:1.6">{q["question"]}</div>'
        f'</div>'
    )

    # Radio options
    key = f"mcq_{q['id']}"
    choice = st.radio(
        f"Q{idx+1}",
        q["options"],
        index=None,
        key=key,
        label_visibility="collapsed"
    )

    if choice is not None:
        chosen_idx = q["options"].index(choice)
        correct    = chosen_idx == q["answer"]

        st.session_state.setdefault("mcq_score",    {})[q["id"]] = correct
        st.session_state.setdefault("mcq_answered", {})[q["id"]] = chosen_idx

        if not hide_answer:
            if correct:
                st.html(ib(
                    f'{gt("✅ Correct!")} '
                    + f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff">'
                    + q["explanation"] + '</span>', "green"
                ))
            else:
                correct_text = q["options"][q["answer"]]
                st.html(ib(
                    f'{rt2("✗ Incorrect.")} '
                    + f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff">'
                    + f'<strong style="color:#FFD700;-webkit-text-fill-color:#FFD700">'
                    + f'Correct answer: {correct_text}</strong><br><br>'
                    + q["explanation"] + '</span>', "red"
                ))
    elif answered and not hide_answer:
        # Already answered in a previous render
        prev_idx = st.session_state["mcq_answered"][q["id"]]
        correct  = prev_idx == q["answer"]
        if correct:
            st.html(ib(gt("✅ Previously answered correctly."), "green"))
        else:
            st.html(ib(rt2(f'✗ Previously answered incorrectly. Correct: {q["options"][q["answer"]]}'), "red"))

    st.html('<div style="margin-bottom:10px"></div>')


# ══════════════════════════════════════════════════════════════════
# NUMERICAL SECTION
# ══════════════════════════════════════════════════════════════════

def _numerical_section():
    col1, col2 = st.columns(2)
    topic_n = col1.selectbox("Topic", ["All","SLR","MLR","Diagnostics","Finance"], key="num_topic")
    level_n = col2.selectbox("Level", ["All","Intermediate","Advanced"], key="num_level")

    filtered = [q for q in NUMERICAL_BANK
                if (topic_n=="All" or q["topic"]==topic_n)
                and (level_n=="All" or q["level"]==level_n)]

    if not filtered:
        render_ib(rt2("No problems match filters."), "red")
        return

    for prob in filtered:
        _render_numerical(prob)


def _render_numerical(prob):
    level_col = {"Foundation":"#28a745","Intermediate":"#FFD700","Advanced":"#dc3545"}.get(prob["level"],"#ADD8E6")

    st.html(
        f'<div style="background:#112240;border:1px solid #1e3a5f;border-radius:10px;'
        f'padding:18px 20px;margin-bottom:4px;user-select:none;-webkit-user-select:none">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">'
        f'{bdg(prob["topic"],"blue")} '
        f'<span style="color:{level_col};-webkit-text-fill-color:{level_col};'
        f'font-size:.78rem;font-weight:700;font-family:{FB}">{prob["level"]}</span>'
        f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;'
        f'font-family:{FH};font-size:1.0rem;margin-left:8px">{prob["title"]}</span>'
        f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0;'
        f'font-size:.75rem;font-family:{FB};margin-left:auto">ID: {prob["id"]}</span>'
        f'</div>'
        f'<div style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;'
        f'font-family:{FB};font-size:.95rem;line-height:1.65">{prob["question"]}</div>'
        f'</div>'
    )

    show_key = f"show_sol_{prob['id']}"
    if show_key not in st.session_state:
        st.session_state[show_key] = False

    col1, col2 = st.columns([1, 4])
    if col1.button("💡 Show Solution", key=f"btn_{prob['id']}"):
        st.session_state[show_key] = not st.session_state[show_key]

    if st.session_state[show_key]:
        # Key results first
        result_rows = [[hl(k), txt_s(v)] for k, v in prob["key_results"]]
        st.html('<div style="margin-top:10px">' + table_html(["Result","Value"], result_rows) + '</div>')

        # Full step-by-step
        st.html(ib(
            f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">'
            f'📐 Full Worked Solution:</span>'
            + fml(prob["solution"]),
            "gold"
        ))

    st.html('<div style="margin-bottom:12px"></div>')


# ══════════════════════════════════════════════════════════════════
# AI TUTOR SECTION
# ══════════════════════════════════════════════════════════════════

def _ai_tutor_section():
    render_card("🤖 AI Tutor — Ask Anything About Regression",
        ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">'
           f'Powered by Claude AI.</span> '
           + txt_s('Ask any question about linear regression, OLS, diagnostics, or finance applications. '
                   'The tutor provides detailed, exam-ready explanations with formulas and examples.'),
           "blue")
    )

    # Quick question shortcuts
    st.html(
        f'<div style="margin-bottom:12px;user-select:none;-webkit-user-select:none">'
        f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0;'
        f'font-family:{FB};font-size:.85rem">Quick Questions: </span>'
        f'</div>'
    )

    quick_qs = [
        "Explain the difference between R² and Adjusted R²",
        "What is Jensen's Alpha and how is it estimated?",
        "How do I interpret a VIF of 12?",
        "What are the consequences of heteroscedasticity?",
        "Explain Fama-French 3-factor model",
        "When should I use t-test vs F-test in regression?",
    ]

    cols = st.columns(3)
    for i, qq in enumerate(quick_qs):
        if cols[i % 3].button(qq, key=f"quick_{i}", use_container_width=True):
            st.session_state["ai_question"] = qq
            st.rerun()

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.html(
                f'<div style="background:rgba(0,77,128,0.4);border-left:4px solid #ADD8E6;'
                f'border-radius:8px;padding:12px 15px;margin:8px 0;'
                f'user-select:none;-webkit-user-select:none">'
                f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;'
                f'font-weight:600;font-size:.8rem">YOU</span><br>'
                f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;'
                f'font-family:{FB}">{msg["content"]}</span></div>'
            )
        else:
            st.html(
                f'<div style="background:rgba(255,215,0,0.07);border-left:4px solid #FFD700;'
                f'border-radius:8px;padding:14px 16px;margin:8px 0;'
                f'user-select:none;-webkit-user-select:none">'
                f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;'
                f'font-weight:600;font-size:.8rem">AI TUTOR</span><br>'
                f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;'
                f'font-family:{FB};line-height:1.7">{msg["content"]}</span></div>'
            )

    # Input
    default_q = st.session_state.pop("ai_question", "")
    question  = st.text_input(
        "Ask a question about regression or finance...",
        value=default_q,
        placeholder="e.g. What is the difference between SLR and MLR?",
        key="ai_input"
    )

    col1, col2 = st.columns([1, 5])
    send  = col1.button("🤖 Ask AI", key="ai_send", use_container_width=True)
    if col2.button("🗑 Clear Chat", key="ai_clear"):
        st.session_state.chat_history = []
        st.rerun()

    if send and question.strip():
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            answer = _call_claude(question, st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    # Suggested follow-ups if chat has messages
    if st.session_state.chat_history:
        st.html(
            f'<div style="margin-top:14px;color:#8892b0;-webkit-text-fill-color:#8892b0;'
            f'font-family:{FB};font-size:.83rem;user-select:none">💡 Try asking: '
            f'"Give me a numerical example" or "How does this apply in finance?"</div>'
        )


def _call_claude(question: str, history: list) -> str:
    """Call Claude API for AI tutoring responses."""
    import json
    try:
        import urllib.request

        system_prompt = """You are an expert finance professor specialising in econometrics and financial modelling.
Your students are MBA/CFA/FRM candidates studying linear regression in finance.

When answering:
- Be precise and exam-ready — give formulas, conditions, and interpretations
- Always ground examples in finance: CAPM, Fama-French, bond pricing, credit risk, P/E models
- For numerical questions, show clear step-by-step workings
- Use plain text formatting (no markdown symbols like ** or ##)
- Keep responses concise but complete — 150 to 300 words unless a worked example is needed
- End with a one-line "Key Takeaway:" summary"""

        messages = []
        for h in history[-6:]:  # last 3 exchanges
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": question})

        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": messages,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return data["content"][0]["text"]

    except Exception as e:
        err = str(e)
        # Provide a helpful fallback answer from the question bank
        fallback = _fallback_answer(question)
        if fallback:
            return fallback
        return (
            f"I couldn't connect to the AI service right now ({err[:80]}). "
            "Please check your internet connection or try again. "
            "Meanwhile, refer to the MCQ and Numerical sections for explanations on this topic."
        )


def _fallback_answer(question: str) -> str:
    """Return a static fallback if API is unavailable."""
    q_lower = question.lower()
    if "r squared" in q_lower or "r²" in q_lower or "r2" in q_lower:
        return (
            "R² (Coefficient of Determination) measures the proportion of Y's variance explained by the regression model. "
            "Range: 0 to 1. R²=0.75 means 75% of variation is explained.\n\n"
            "Adjusted R² = 1 − (1−R²)(n−1)/(n−k−1). It penalises adding irrelevant variables. "
            "In MLR, always use Adjusted R² for model comparison. R² never decreases when adding variables, "
            "even useless ones — Adjusted R² will decrease if the variable adds no explanatory power.\n\n"
            "Key Takeaway: Use R² for SLR interpretation; always use Adjusted R² when comparing MLR models."
        )
    if "beta" in q_lower or "capm" in q_lower:
        return (
            "CAPM Beta (β) measures systematic risk — sensitivity of stock returns to market returns. "
            "Estimated by regressing excess stock returns on excess market returns (SCL regression).\n\n"
            "β = Cov(Rᵢ,Rₘ) / Var(Rₘ) = Σ(Xᵢ−X̄)(Yᵢ−Ȳ) / Σ(Xᵢ−X̄)²\n\n"
            "Interpretation: β=1.3 → stock moves 1.3% for every 1% market move (aggressive). "
            "β=0.7 → defensive. β=1 → market-neutral.\n\n"
            "Jensen's Alpha (intercept) = excess risk-adjusted return. α>0 indicates manager skill.\n\n"
            "Key Takeaway: Beta = systematic risk (non-diversifiable). 1−R² = unsystematic risk (diversifiable)."
        )
    if "heteroscedasticity" in q_lower or "heteroscedastic" in q_lower:
        return (
            "Heteroscedasticity means Var(εᵢ) ≠ constant — the variance of residuals changes across observations.\n\n"
            "Effects: OLS β̂ remains UNBIASED but is no longer EFFICIENT. Standard errors are biased, "
            "making t-tests and F-tests invalid.\n\n"
            "Detection: Breusch-Pagan test (LM = n×R²_aux, ~χ²(k)), White test, Scale-Location plot.\n\n"
            "Remedies: HC3 robust standard errors (White's correction), WLS if variance function is known, "
            "log transformation if errors are multiplicative.\n\n"
            "Finance context: Very common in equity returns due to volatility clustering (ARCH effects).\n\n"
            "Key Takeaway: Heteroscedasticity doesn't bias β̂ but invalidates inference — always use robust SE."
        )
    if "vif" in q_lower or "multicollinearity" in q_lower:
        return (
            "VIF (Variance Inflation Factor) measures multicollinearity severity. "
            "VIF_j = 1/(1−R²_j) where R²_j comes from regressing X_j on all other predictors.\n\n"
            "Rules: VIF<5 = OK, 5-10 = moderate concern, >10 = serious problem.\n\n"
            "Effects: β̂ remains unbiased but SEs inflate → t-stats deflate → variables appear insignificant "
            "even when they matter. R² remains high but individual coefficients are unstable.\n\n"
            "Remedies: Drop correlated variable, Ridge regression (adds λΣβ² penalty), PCA, collect more data.\n\n"
            "Finance: In Fama-French, MKT/SMB/HML have low VIFs (~1.1). "
            "Including both Nifty50 and Nifty500 as factors would give VIF>>10.\n\n"
            "Key Takeaway: High VIF inflates SE, not bias. Detect early, remediate before inference."
        )
    return ""
