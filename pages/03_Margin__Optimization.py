# pages/03_Margin__Optimization.py
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(page_title="Margin Optimization", page_icon="📈", layout="wide")
st.title("Rivard Labs - Innocap: Margin Requirement Estimator")

# ----------------------------
# EDIT THESE PLACEHOLDER WEIGHTS LATER
# ----------------------------
COEFFS = {
    "intercept": 0.0,  # USD
    "weights_num": {
        # core signal ~5x VaR
        "var_1d_99_usd": 5.00,         # <-- main driver
        # tiny stabilizers (feel free to zero out)
        "gross_notional_ex_cash_usd": 0.00,
        "long_notional_ex_cash_usd":  0.00,
        "short_notional_ex_cash_usd": 0.00,
        "vol_exante_20d":             0.0,   # per % point
        "vol_realized_20d":           0.0,   # per % point
        # exposures (default 0 so it’s effectively just 5x VaR)
        "exposure_equity_usd":        0.00,
        "exposure_fi_credit_usd":     0.00,
        "exposure_fi_govt_usd":       0.00,
        "exposure_other_usd":         0.00,
        "exposure_future_usd":        0.00,
        "exposure_swap_usd":          0.00,
        "exposure_options_usd":       0.00,
        "exposure_cfd_usd":           0.00,
        "exposure_instr_other_usd":   0.00,
        # stress add-on (negative PnL increases margin if you want; keep 0 to ignore)
        "covid_equity_pnl_usd":       0.00,  # set to e.g. -0.5 to add back losses
        "covid_fi_credit_pnl_usd":    0.00,
        "covid_fi_govt_pnl_usd":      0.00,
    },
    "weights_cat": {
        # one-hot style bumps (flat USD adders), edit/delete as needed
        "strategy_Equity LS": 0.0,
        "strategy_Macro":     0.0,
        "strategy_Credit":    0.0,
        "strategy_Other":     0.0,
        "pb_id_PB1":          0.0,
    },
}
# ----------------------------

def score_margin(features: dict) -> float:
    """Pure linear scorer using COEFFS in raw units (USD / %)."""
    y = float(COEFFS.get("intercept", 0.0))
    WN = COEFFS.get("weights_num", {})
    for k, w in WN.items():
        if k in features and features[k] is not None and not (isinstance(features[k], float) and np.isnan(features[k])):
            y += float(w) * float(features[k])
    WC = COEFFS.get("weights_cat", {})
    for name, w in WC.items():
        base, val = name.split("_", 1)
        if str(features.get(base, "")) == val:
            y += float(w)
    return max(0.0, float(y))

col1, col2 = st.columns(2)
strategy = col1.selectbox("Strategy", ["Equity LS","Macro","Credit","Other"])
pb_id    = col2.text_input("Prime Broker ID", "PB1")

st.subheader("Core Risk Inputs")
var_99   = st.number_input("VaR 1d 99% (USD)", value=1_000_000.0, step=1_000.0, min_value=0.0)
gross    = st.number_input("Gross Notional ex Cash (USD)", value=10_000_000.0, step=10_000.0, min_value=0.0)
longs    = st.number_input("Long Notional ex Cash (USD)", value=6_000_000.0,  step=10_000.0, min_value=0.0)
shorts   = st.number_input("Short Notional ex Cash (USD)", value=4_000_000.0, step=10_000.0, min_value=0.0)
vol_ex   = st.number_input("Vol (ex-ante, 20d, %)", value=15.0, step=0.1, min_value=0.0)
vol_rl   = st.number_input("Vol (realized, 20d, %)", value=12.0, step=0.1, min_value=0.0)

st.subheader("Exposure Buckets (USD)")
c1, c2 = st.columns(2)
equity      = c1.number_input("Equity",   value=5_000_000.0, step=10_000.0, min_value=0.0)
fi_credit   = c1.number_input("FI Credit",value=2_000_000.0, step=10_000.0, min_value=0.0)
fi_gov      = c1.number_input("FI Gov",   value=1_000_000.0, step=10_000.0, min_value=0.0)
other       = c1.number_input("Other",    value=2_000_000.0, step=10_000.0, min_value=0.0)
future      = c2.number_input("Futures",  value=1_000_000.0, step=10_000.0, min_value=0.0)
swap        = c2.number_input("Swaps",    value=500_000.0,  step=10_000.0, min_value=0.0)
options     = c2.number_input("Options",  value=500_000.0,  step=10_000.0, min_value=0.0)
cfd         = c2.number_input("CFDs",     value=250_000.0,  step=10_000.0, min_value=0.0)
instr_other = c2.number_input("Instr Other", value=250_000.0, step=10_000.0, min_value=0.0)

st.subheader("Stress (optional, USD)")
cov_eq  = st.number_input("COVID Equity PnL",    value=-200_000.0, step=1_000.0)
cov_cr  = st.number_input("COVID FI Credit PnL", value=-100_000.0, step=1_000.0)
cov_gov = st.number_input("COVID FI Gov PnL",    value=  50_000.0, step=1_000.0)

if st.button("Estimate Margin"):
    features = {
        "strategy": strategy, "pb_id": pb_id,
        "var_1d_99_usd": var_99, "var_1d_95_usd": np.nan,
        "gross_notional_ex_cash_usd": gross,
        "long_notional_ex_cash_usd": longs,
        "short_notional_ex_cash_usd": shorts,
        "vol_exante_20d": vol_ex, "vol_realized_20d": vol_rl,
        "exposure_equity_usd": equity, "exposure_fi_credit_usd": fi_credit,
        "exposure_fi_govt_usd": fi_gov, "exposure_other_usd": other,
        "exposure_future_usd": future, "exposure_swap_usd": swap,
        "exposure_options_usd": options, "exposure_cfd_usd": cfd, "exposure_instr_other_usd": instr_other,
        "covid_equity_pnl_usd": cov_eq, "covid_fi_credit_pnl_usd": cov_cr, "covid_fi_govt_pnl_usd": cov_gov,
    }

    margin_usd = score_margin(features)
    pct_gross  = (margin_usd / gross * 100.0) if gross > 0 else float("nan")
    mult_var   = (margin_usd / var_99)        if var_99 > 0 else float("nan")

    st.success(
        f"Estimated Margin: **${margin_usd:,.0f}**  \n"
        f"• As % of Gross: **{pct_gross:,.2f}%**  \n"
        f"• Multiple of VaR(1d 99%): **{mult_var:,.2f}×**"
    )

st.caption("Using a simple linear equation with placeholder weights (default ≈ 5×VaR). Replace values in COEFFS when your real coefficients are ready.")
