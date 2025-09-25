# File: pages/02_ETF_Mean_Reversion.py
# Purpose: Excel-driven ETF table (Yahoo-only), Top-10 below selected MA,
#          count below selected MA, and Method #1 empirical mean-reversion probability.

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ETF Mean Reversion", layout="wide")
st.title("ETF Mean Reversion")

# --- Do nothing until user explicitly enables + runs ---
enable = st.toggle("Enable this page", value=False, help="Prevents any heavy work until checked.")
run = st.button("Run backtest")
if not (enable and run):
    st.info("Toggle **Enable this page** and click **Run backtest** to load data.")
    st.stop()

# --- DIAGNOSTIC (remove later) ---
import os, sys, hashlib, pathlib, streamlit as st
VERSION = "ETF page :: diag v5-top10plot"
this_file = pathlib.Path(__file__).resolve()
sha = hashlib.sha1(open(this_file, "rb").read()).hexdigest()[:12]
st.info(f"Running: `{this_file}` • Python: `{sys.executable}` • SHA: {sha} • {VERSION}")
# ---------------------------------

# ------------------------------ Config ------------------------------
MASTER_PATHS = [
    Path("data") / "ETF project.xlsm",
    Path("ETF project.xlsm"),
]
MASTER_SHEET = "Tickers"

DEFAULT_HIST_PERIOD = "3y"  # for empirical stats
SUPPORTED_PERIODS = ["2y", "3y", "5y"]

# ------------------------------ Helpers ------------------------------
def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def _series_or_blank(df: pd.DataFrame, colname: Optional[str]) -> pd.Series:
    if colname:
        return df[colname].fillna("").astype(str).str.strip()
    return pd.Series([""] * len(df), index=df.index, dtype="object")

def _series_or_nan(df: pd.DataFrame, colname: Optional[str]) -> pd.Series:
    if colname:
        return df[colname]
    return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")

def _fmt_price(x):
    return "" if pd.isna(x) else f"{x:.2f}"

def _fmt_pct_str(x):
    return "" if pd.isna(x) else f"{x:.2f}%"

# ------------------------- Excel Master Loader -------------------------
@st.cache_data(show_spinner=False)
def _load_master_from_path(src: Path) -> pd.DataFrame:
    df = pd.read_excel(src, sheet_name=MASTER_SHEET, engine="openpyxl")
    return _normalize_master(df)

@st.cache_data(show_spinner=False)
def _load_master_from_bytes(content: bytes) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(content), sheet_name=MASTER_SHEET, engine="openpyxl")
    return _normalize_master(df)

def _normalize_master(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    def _col_at(idx: int) -> Optional[str]:
        return df.columns[idx] if 0 <= idx < len(df.columns) else None

    col_map = {
        "ticker": None,        # B
        "fund_name": None,     # C
        "hyperlink": None,     # D
        "brand": None,         # E
        "expense_ratio": None, # F
        "focus": None,         # G
        "region": None,        # H
    }

    for c in df.columns:
        cl = c.lower()
        if col_map["ticker"] is None and cl in ("ticker", "tickers", "symbol"):
            col_map["ticker"] = c
        elif col_map["fund_name"] is None and cl in ("fund name", "fullname", "full name", "name", "fund"):
            col_map["fund_name"] = c
        elif col_map["hyperlink"] is None and cl in ("hyperlink", "hyperlinks", "url", "link"):
            col_map["hyperlink"] = c
        elif col_map["brand"] is None and cl in ("brand", "provider", "company", "issuer", "etf name"):
            col_map["brand"] = c
        elif col_map["expense_ratio"] is None and ("expense" in cl and "ratio" in cl):
            col_map["expense_ratio"] = c
        elif col_map["focus"] is None and cl in ("focus", "category", "sector"):
            col_map["focus"] = c
        elif col_map["region"] is None and "region" in cl:
            col_map["region"] = c

    col_map["ticker"]        = col_map["ticker"]        or _col_at(1)  # B
    col_map["fund_name"]     = col_map["fund_name"]     or _col_at(2)  # C
    col_map["hyperlink"]     = col_map["hyperlink"]     or _col_at(3)  # D
    col_map["brand"]         = col_map["brand"]         or _col_at(4)  # E
    col_map["expense_ratio"] = col_map["expense_ratio"] or _col_at(5)  # F
    col_map["focus"]         = col_map["focus"]         or _col_at(6)  # G
    col_map["region"]        = col_map["region"]        or _col_at(7)  # H

    required_for_min = []
    if col_map["ticker"] is None:
        required_for_min.append("Ticker (column B)")
    if (col_map["fund_name"] is None) and (col_map["brand"] is None):
        required_for_min.append("Fund Name (column C) or Brand (column E)")
    if required_for_min:
        raise ValueError(
            f"Missing required columns in sheet '{MASTER_SHEET}': {', '.join(required_for_min)}"
        )

    tick_raw = df[col_map["ticker"]]
    ticker_ser = tick_raw.astype(str).str.strip().str.upper()

    name_ser = (
        _series_or_blank(df, col_map["fund_name"])
        if col_map["fund_name"] else _series_or_blank(df, col_map["brand"])
    )

    url_ser    = _series_or_blank(df, col_map["hyperlink"])
    brand_ser  = _series_or_blank(df, col_map["brand"])
    expense_raw = _series_or_nan(df, col_map["expense_ratio"])
    focus_ser  = _series_or_blank(df, col_map["focus"]).str.title()
    region_ser = _series_or_blank(df, col_map["region"])

    out = pd.DataFrame({
        "Ticker": ticker_ser,
        "URL": url_ser,
        "Name": name_ser,
        "Brand": brand_ser,
        "Expense Ratio Raw": expense_raw,
        "Category": focus_ser,
        "Region": region_ser,
    })

    name_blank = out["Name"].eq("") | out["Name"].isna()
    out.loc[name_blank, "Name"] = out.loc[name_blank, "Brand"]
    name_blank = out["Name"].eq("") | out["Name"].isna()
    out.loc[name_blank, "Name"] = out.loc[name_blank, "Ticker"]

    def _parse_expense(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s == "":
            return np.nan
        m_bps = re.search(r"([\d\.]+)\s*bp", s)
        if m_bps:
            try:
                bps = float(m_bps.group(1))
                return bps / 100.0  # bps → %
            except Exception:
                pass
        if s.endswith("%"):
            try:
                return float(s.replace("%", "").strip())
            except Exception:
                pass
        try:
            val = float(s)
            return val * 100.0 if val < 1 else val
        except Exception:
            return np.nan

    out["Expense Ratio (%)"] = out["Expense Ratio Raw"].apply(_parse_expense).round(2)
    out.drop(columns=["Expense Ratio Raw"], inplace=True)

    out = out[~out["Ticker"].isna() & (out["Ticker"].str.strip() != "")]
    out = out.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return out

# ----------------------------- Yahoo Snapshot -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_prices_yahoo_snapshot(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch a 2y snapshot from Yahoo and compute Price, MA50/MA200, % spreads,
    plus NAV, signed premium/discount to NAV, Yield, Expense Ratio (net), Net Assets.
    Always returns a DataFrame with expected columns (even if empty).
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance") from e

    def _pct_from_yahoo(x):
        if x is None or (isinstance(x, float) and (np.isnan(x))):
            return None
        try:
            v = float(x)
            if 0 <= v <= 1:
                return v * 100.0
            return v
        except Exception:
            return None

    recs = []
    for t in tickers:
        t = (t or "").strip().upper()
        if not t:
            continue
        price = ma50 = ma200 = pct_vs50 = pct_vs200 = None

        nav = None
        prem_disc_nav = None
        trailing_yield_pct = None
        expense_ratio_yahoo_pct = None
        net_assets_usd = None

        try:
            yt = yf.Ticker(t)
            hist = yt.history(period="2y", auto_adjust=False)
            if isinstance(hist.index, pd.DatetimeIndex) and hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
            if not hist.empty and "Close" in hist:
                close = hist["Close"].dropna()
                if len(close) > 0:
                    price = float(close.iloc[-1])
                    ma50_s = close.rolling(50, min_periods=50).mean()
                    ma200_s = close.rolling(200, min_periods=200).mean()
                    ma50 = float(ma50_s.iloc[-1]) if not pd.isna(ma50_s.iloc[-1]) else None
                    ma200 = float(ma200_s.iloc[-1]) if not pd.isna(ma200_s.iloc[-1]) else None
                    if price is not None and ma50:
                        pct_vs50 = (price - ma50) / ma50 * 100.0
                    if price is not None and ma200:
                        pct_vs200 = (price - ma200) / ma200 * 100.0

            try:
                info = yt.info or {}
            except Exception:
                info = {}

            nav_raw = info.get("navPrice", None)
            try:
                if nav_raw is not None:
                    nav = float(nav_raw)
            except Exception:
                nav = None

            if nav and price:
                prem_disc_nav = (price - nav) / nav * 100.0

            y_raw = info.get("yield", None) or info.get("trailingAnnualDividendYield", None)
            trailing_yield_pct = _pct_from_yahoo(y_raw)

            er_raw = info.get("annualReportExpenseRatio", None) or info.get("feesExpensesInvestment", None)
            expense_ratio_yahoo_pct = _pct_from_yahoo(er_raw)

            try:
                net_assets_usd = float(info.get("totalAssets", None)) if info.get("totalAssets", None) is not None else None
            except Exception:
                net_assets_usd = None

        except Exception:
            pass

        recs.append(
            {
                "Ticker": t,
                "Price": price,
                "MA50": ma50,
                "MA200": ma200,
                "Price vs 50d %": pct_vs50,
                "Price vs 200d %": pct_vs200,
                "NAV": nav,
                "Prem/Disc to NAV %": prem_disc_nav,
                "Yield % (Yahoo)": trailing_yield_pct,
                "Expense Ratio (Yahoo %)": expense_ratio_yahoo_pct,
                "Net Assets (USD)": net_assets_usd,
            }
        )

    expected_cols = [
        "Ticker", "Price", "MA50", "MA200", "Price vs 50d %", "Price vs 200d %",
        "NAV", "Prem/Disc to NAV %", "Yield % (Yahoo)", "Expense Ratio (Yahoo %)", "Net Assets (USD)",
    ]
    df = pd.DataFrame.from_records(recs)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object" if col == "Ticker" else "float64")
    return df[expected_cols]

@st.cache_data(show_spinner=False, ttl=7200)
def _fetch_history_yahoo(tickers: List[str], period: str = DEFAULT_HIST_PERIOD) -> Dict[str, pd.Series]:
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance") from e

    if not tickers:
        return {}

    data = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    out: Dict[str, pd.Series] = {}

    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                ser = data[(t, "Close")].dropna()
                if isinstance(ser.index, pd.DatetimeIndex) and ser.index.tz is not None:
                    ser.index = ser.index.tz_localize(None)
                ser.name = t
                out[t] = ser
            except Exception:
                continue
    else:
        try:
            ser = data["Close"].dropna()
            if isinstance(ser.index, pd.DatetimeIndex) and ser.index.tz is not None:
                ser.index = ser.index.tz_localize(None)
            first_ticker = tickers[0] if tickers else "TICKER"
            ser.name = first_ticker
            out[first_ticker] = ser
        except Exception:
            pass
    return out

# ----------------- Empirical Probability (Method #1, vectorized) -----------------
def compute_empirical_reversion_prob(
    close: pd.Series,
    ma_window: int,
    H: int,
    eps: float,
    bin_edges: np.ndarray,
    min_samples_per_bin: int = 30
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if close is None or len(close) < (ma_window + H + 20):
        return (None, None, None)

    mu = close.rolling(ma_window, min_periods=ma_window).mean()
    sd = close.rolling(ma_window, min_periods=ma_window).std(ddof=0)
    valid = (~mu.isna()) & (sd > 0)
    if valid.sum() < (ma_window + H):
        return (None, None, None)

    z = (close - mu) / sd
    z = z.where(np.isfinite(z))

    abs_z = z.abs()
    fwd_min = abs_z.shift(-1).rolling(window=H, min_periods=H).min()
    hit = (fwd_min <= eps).astype(float).where(fwd_min.notna())

    z_vals = z.to_numpy()
    hit_vals = hit.to_numpy()
    mask = np.isfinite(z_vals) & np.isfinite(hit_vals)

    if mask.sum() < min_samples_per_bin:
        return (None, float(z.iloc[-1]) if np.isfinite(z.iloc[-1]) else None, None)

    bins = np.digitize(z_vals, bin_edges)
    probs_by_bin: Dict[int, float] = {}
    counts_by_bin: Dict[int, int] = {}

    for b in range(1, len(bin_edges) + 1):
        idx = mask & (bins == b)
        cnt = int(idx.sum())
        if cnt >= min_samples_per_bin:
            p = float(np.nanmean(hit_vals[idx]))
            probs_by_bin[b] = p
            counts_by_bin[b] = cnt

    current_z = float(z.iloc[-1]) if np.isfinite(z.iloc[-1]) else None
    if current_z is None:
        return (None, None, None)
    current_bin = int(np.digitize([current_z], bin_edges)[0])

    p_use = None
    coverage = None
    for b in [current_bin, current_bin - 1, current_bin + 1, current_bin - 2, current_bin + 2]:
        if b in probs_by_bin:
            p_use = probs_by_bin[b]
            total = int(np.isfinite(hit_vals).sum())
            coverage = counts_by_bin[b] / max(1, total)
            break

    if p_use is None:
        return (None, current_z, None)

    return (100.0 * p_use, current_z, 100.0 * (coverage if coverage is not None else np.nan))

# ---------------------------- UI: Controls ----------------------------
with st.sidebar:
    st.header("Signal Settings")
    ma_for_signal = st.selectbox(
        "Mean for Statistical Signal",
        options=[50, 200],
        index=0,
        help="Applies to the empirical probability computation AND the summary/top-10 lists."
    )
    H = st.number_input("Horizon H (trading days)", min_value=5, max_value=60, value=10, step=1)
    eps = st.number_input("Band width ε (z-units)", min_value=0.05, max_value=1.00, value=0.20, step=0.05, format="%.2f")
    bin_width = st.selectbox("Z-bin width", options=[0.25, 0.5, 1.0], index=1)
    min_bin = st.number_input("Min samples per bin", min_value=10, max_value=200, value=30, step=5)
    hist_period = st.selectbox("Historical period (Yahoo)", options=SUPPORTED_PERIODS, index=SUPPORTED_PERIODS.index(DEFAULT_HIST_PERIOD))

# ---------------------------- Load Master ----------------------------
src_path = _first_existing(MASTER_PATHS)
uploaded = None
if not src_path:
    with st.expander("Excel master not found — upload it here"):
        uploaded = st.file_uploader("Upload 'ETF project.xlsm' (sheet 'Tickers')", type=["xlsm", "xlsx"], accept_multiple_files=False)

base: Optional[pd.DataFrame] = None
try:
    if src_path:
        base = _load_master_from_path(src_path)
        st.caption(f"Loaded {len(base)} tickers from Excel master at **{src_path}** (sheet '{MASTER_SHEET}').")
    elif uploaded is not None:
        base = _load_master_from_bytes(uploaded.getvalue())
        st.caption(f"Loaded {len(base)} tickers from uploaded file (sheet '{MASTER_SHEET}').")
    else:
        st.error("Could not find 'ETF project.xlsm'. Place it in ./data or project root, or upload it above.")
        st.stop()
except Exception as e:
    st.error(f"Failed to load master: {e}")
    st.stop()

if base is None or base.empty:
    st.error("No tickers were loaded from the Excel master.")
    st.stop()

tickers: List[str] = base["Ticker"].tolist()

# ---------------------------- Snapshot Fetch (Yahoo only) ----------------------------
try:
    px = _fetch_prices_yahoo_snapshot(tickers)
except Exception as e:
    st.warning(f"Snapshot failed: {e}")
    px = pd.DataFrame(columns=[
        "Ticker","Price","MA50","MA200","Price vs 50d %","Price vs 200d %",
        "NAV","Prem/Disc to NAV %","Yield % (Yahoo)","Expense Ratio (Yahoo %)","Net Assets (USD)"
    ])

try:
    hist_map = _fetch_history_yahoo(tickers, period=hist_period)
except Exception as e:
    st.warning(f"History download failed: {e}")
    hist_map = {}


# ---------------------------- Historical Fetch ----------------------------
hist_map = _fetch_history_yahoo(tickers, period=hist_period)

# Empirical probability per ticker
bin_edges = np.arange(-3.0, 3.0 + float(bin_width), float(bin_width))
results = []
for t in tickers:
    ser = hist_map.get(t)
    if ser is None or ser.empty:
        results.append((None, None, None))
        continue
    p, z_now, cov = compute_empirical_reversion_prob(
        close=ser, ma_window=int(ma_for_signal), H=int(H), eps=float(eps),
        bin_edges=bin_edges, min_samples_per_bin=int(min_bin)
    )
    results.append((p, z_now, cov))

prob_df = pd.DataFrame(results, columns=["Stat Buy Prob (%)", "Current Z", "Empirical Coverage (%)"])
prob_df["Ticker"] = tickers
df = df.merge(prob_df, on="Ticker", how="left")

# ---------------------------- Summary + Top 10 (dynamic by selected MA) ----------------------------
selected_pct_col = "Price vs 50d %" if ma_for_signal == 50 else "Price vs 200d %"

tmp = df.copy()
tmp["below_sel"] = tmp[selected_pct_col].apply(lambda v: (not pd.isna(v)) and v < 0)
tmp["below_sel_abs"] = tmp[selected_pct_col].apply(lambda v: abs(v) if (not pd.isna(v) and v < 0) else 0.0)
top10 = tmp[tmp["below_sel"]].sort_values("below_sel_abs", ascending=False).head(10)

count_below_sel = int(((~df[selected_pct_col].isna()) & (df[selected_pct_col] < 0)).sum())

st.subheader("Summary")
c1, c2 = st.columns(2)
with c1:
    st.metric(f"Count below {ma_for_signal}-day MA", f"{count_below_sel}")
with c2:
    st.caption(
        f"Statistical Buy Prob uses Method #1 on Yahoo history over {hist_period}, "
        f"MA={ma_for_signal}, H={H}, ε={eps} (vectorized)."
    )

# ---------- NEW: split Top 10 into left list (hyperlinked tickers) and right scatter plot ----------
st.subheader(f"Top 10 • Potential Mean Reversion (below {ma_for_signal}-day MA)")
if top10.empty:
    st.write(f"No entries currently below their {ma_for_signal}-day MA.")
else:
    left, right = st.columns([1, 1])

    # Left: hyperlinked list
    with left:
        for _, r in top10.iterrows():
            ptxt = "" if pd.isna(r["Stat Buy Prob (%)"]) else f" • Stat Prob: {r['Stat Buy Prob (%)']:.1f}%"
            # make ticker a markdown link when URL exists
            if pd.notna(r.get("URL")) and str(r.get("URL")).strip():
                ticker_md = f"[{r['Ticker']}]({r['URL']})"
            else:
                ticker_md = r["Ticker"]
            st.markdown(
                f"- **{r['Name']} ({ticker_md})** — {abs(r[selected_pct_col]):.2f}% below {ma_for_signal}d MA{ptxt}"
            )

    # Right: scatter plot of depth vs probability with ticker labels
    with right:
        import matplotlib.pyplot as plt

        plot_df = top10.copy()
        plot_df["Depth Below MA %"] = plot_df[selected_pct_col].abs()
        plot_df["Prob %"] = plot_df["Stat Buy Prob (%)"]

        plot_df = plot_df[plot_df["Prob %"].notna() & plot_df["Depth Below MA %"].notna()]
        if plot_df.empty:
            st.info("Not enough data to plot the scatter (need valid probability and depth for at least one ticker).")
        else:
            fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
            ax.scatter(plot_df["Depth Below MA %"], plot_df["Prob %"])

            # annotate each point with ticker
            for _, row in plot_df.iterrows():
                ax.annotate(
                    row["Ticker"],
                    (row["Depth Below MA %"], row["Prob %"]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=8
                )

            ax.set_xlabel(f"Depth below {ma_for_signal}-day MA (%)")
            ax.set_ylabel(f"Stat Buy Prob (%) (H={H}, ε={eps})")
            ax.set_title("Top 10: Mean-Reversion Setup")
            # neat limits
            x_max = float(plot_df["Depth Below MA %"].max()) if not plot_df["Depth Below MA %"].empty else 0.0
            y_max = float(plot_df["Prob %"].max()) if not plot_df["Prob %"].empty else 0.0
            ax.set_xlim(left=0, right=max(5.0, x_max * 1.15))
            ax.set_ylim(bottom=0, top=max(50.0, y_max * 1.10))
            ax.grid(True, alpha=0.3)

            st.pyplot(fig, clear_figure=True)
# ----------------------------------------------------------------------------------------------------

st.divider()
st.subheader("All Instruments • Sort columns, search, and download")

# ---------------------------- Table Data (keep numerics for proper sorting) ----------------------------
display = {
    "Name": df["Name"],
    "Ticker": df["Ticker"],
    "Category": df["Category"],
    "Region": df["Region"],
    "Expense Ratio (%)": df["Expense Ratio (%)"],
    "Expense Ratio (net, Yahoo %)": df["Expense Ratio (Yahoo %)"],
    "Yield % (Yahoo)": df["Yield % (Yahoo)"],
    "Net Assets (USD)": df["Net Assets (USD)"],
    "NAV": df["NAV"],
    "Prem/Disc to NAV %": df["Prem/Disc to NAV %"],
    "Price": df["Price"],
    "50d MA": df["MA50"],
    "200d MA": df["MA200"],
    "Price vs 50d %": df["Price vs 50d %"],
    "Price vs 200d %": df["Price vs 200d %"],
    f"Stat Buy Prob (%) (to MA{ma_for_signal} in {H}d, ε={eps})": df["Stat Buy Prob (%)"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}%"),
    "URL": df["URL"],
}
display_cols = pd.DataFrame(display)

# Auto-hide Yahoo-only columns if completely empty
for maybe_empty in ["Expense Ratio (net, Yahoo %)", "Yield % (Yahoo)"]:
    if maybe_empty in display_cols and display_cols[maybe_empty].notna().sum() == 0:
        display_cols.drop(columns=[maybe_empty], inplace=True)

# Reorder UI — 'Name' column is pinned
cols = list(display_cols.columns)
pinned = ["Name"]
reorderable = [c for c in cols if c not in pinned]
new_order = st.multiselect(
    "Reorder (the 'Name' column is pinned and always shown)",
    options=reorderable,
    default=reorderable,
)
final_cols = pinned + new_order

# Column formatting (keeps numeric sorting)
colcfg = {}
for c in ["Price", "50d MA", "200d MA", "NAV"]:
    if c in display_cols.columns: colcfg[c] = st.column_config.NumberColumn(format="%.2f")
for c in ["Price vs 50d %", "Price vs 200d %", "Prem/Disc to NAV %", "Expense Ratio (%)", "Expense Ratio (net, Yahoo %)", "Yield % (Yahoo)"]:
    if c in display_cols.columns: colcfg[c] = st.column_config.NumberColumn(format="%.2f%%")
if "Net Assets (USD)" in display_cols.columns:
    colcfg["Net Assets (USD)"] = st.column_config.NumberColumn(format="$%,.0f")

st.dataframe(display_cols[final_cols], use_container_width=True, column_config=colcfg)

csv_bytes = display_cols[final_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download table (CSV)", data=csv_bytes, file_name="etf_metrics.csv", mime="text/csv")

st.caption(
    "Empirical probability (Method #1): bucket historical z-scores vs the selected MA; "
    "for each bucket, compute the fraction of cases that touch |z| ≤ ε within H days. "
    "If the current bin is sparse, we back off to neighboring bins."
)
