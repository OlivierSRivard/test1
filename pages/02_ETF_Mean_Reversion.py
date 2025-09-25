# File: pages/02_ETF_Mean_Reversion.py
# Purpose: ETF mean-reversion dashboard using a prewarmed Parquet master + Parquet price cache.
# - Reads master from data/cache/ETF_master_copy.parquet (fast; no Excel required)
# - Loads cached history from data/cache/price_history.parquet (ideally 5y prewarmed)
# - Appends only a tiny recent window from Yahoo (for freshness)
# - Lets you ADD tickers (ALL CAPS) after pressing "Run backtest" and includes them in the run
# - Robust probability calc: merges sparse bins and falls back to unconditional rate

from __future__ import annotations

import io
import re
import sys
import hashlib
import pathlib
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional: gate by auth flag in session_state
if not st.session_state.get("is_authed"):
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {display:none !important;}
            [data-testid="stSidebarNav"] {display:none !important;}
            [data-testid="collapsedControl"] {display:none !important;}
        </style>
    """, unsafe_allow_html=True)
    try:
        st.switch_page("Home.py")
    except Exception:
        st.stop()

# ------------------------------ Page & Run Gate ------------------------------
st.set_page_config(page_title="ETF Mean Reversion", layout="wide")
st.title("ETF Mean Reversion")

# Persist run state so you can add tickers post-run
if "run_ok" not in st.session_state:
    st.session_state["run_ok"] = False

enable = st.toggle("Enable this page", value=False, help="Prevents any heavy work until checked.")
if st.button("Run backtest"):
    st.session_state["run_ok"] = True

if not (enable and st.session_state["run_ok"]):
    st.info("Toggle **Enable this page** and click **Run backtest** to load data.")
    st.stop()

# ------------------------------ Config ------------------------------
MASTER_PARQUET = Path("data/cache/ETF_master_copy.parquet")  # prewarmed by tools/prewarm_data.py
CACHE_PARQUET  = Path("data/cache/price_history.parquet")    # prewarmed price history (ideally ~5y)
DEFAULT_HIST_PERIOD = "3y"                                   # only used when cache is missing
SUPPORTED_PERIODS   = ["2y", "3y", "5y"]

# ------------------------------ Helpers ------------------------------
def _series_or_blank(df: pd.DataFrame, colname: Optional[str]) -> pd.Series:
    if colname:
        return df[colname].fillna("").astype(str).str.strip()
    return pd.Series([""] * len(df), index=df.index, dtype="object")

def _series_or_nan(df: pd.DataFrame, colname: Optional[str]) -> pd.Series:
    if colname:
        return df[colname]
    return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")

def _normalize_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a master DataFrame into: Ticker, URL, Name, Brand, Expense Ratio (%), Category, Region.
    Works for the prewarmed parquet (already tidy) or a raw sheet-like DataFrame.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    def _col_at(idx: int) -> Optional[str]:
        return df.columns[idx] if 0 <= idx < len(df.columns) else None

    col_map = {
        "ticker": None,
        "fund_name": None,
        "hyperlink": None,
        "brand": None,
        "expense_ratio": None,
        "focus": None,
        "region": None,
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

    # Fallback positional guesses (if user gave non-standard headers)
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
        raise ValueError("Missing required columns in master parquet: " + ", ".join(required_for_min))

    tick_raw   = df[col_map["ticker"]]
    ticker_ser = tick_raw.astype(str).str.strip().str.upper()
    name_ser   = _series_or_blank(df, col_map["fund_name"]) if col_map["fund_name"] else _series_or_blank(df, col_map["brand"])
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

    # Fill missing Name
    name_blank = out["Name"].eq("") | out["Name"].isna()
    out.loc[name_blank, "Name"] = out.loc[name_blank, "Brand"]
    name_blank = out["Name"].eq("") | out["Name"].isna()
    out.loc[name_blank, "Name"] = out.loc[name_blank, "Ticker"]

    # Parse Expense Ratio to percent
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

    # Clean + dedupe
    out = out[~out["Ticker"].isna() & (out["Ticker"].str.strip() != "")]
    out = out.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return out

# ------------------------------ Master Loader (Parquet only) ------------------------------
@st.cache_data(show_spinner=False)
def _load_master_from_parquet(src: Path) -> pd.DataFrame:
    if not src.exists():
        raise FileNotFoundError(f"Master parquet not found: {src}. Run tools/prewarm_data.py first.")
    raw = pd.read_parquet(src)
    return _normalize_master(raw)

# ---------------------------- Cached history loaders ----------------------------
@st.cache_data(show_spinner=False, ttl=900)
def _load_cached_history(parquet_path: Path, tickers: List[str]) -> Dict[str, pd.Series]:
    """
    Load cached (Date, Ticker, Close) from Parquet and return {ticker: Close Series}.
    """
    if not parquet_path.exists():
        return {}
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return {}
    df = df[df["Ticker"].isin(tickers)].copy()
    out: Dict[str, pd.Series] = {}
    for t, g in df.groupby("Ticker"):
        s = g.sort_values("Date").set_index("Date")["Close"].astype(float)
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s.name = t
        out[t] = s
    return out

def _append_latest_5d(hist_map: Dict[str, pd.Series], tickers: List[str]) -> Dict[str, pd.Series]:
    """
    Download a small recent window (~10 calendar days) and append to existing series, de-duplicated.
    """
    try:
        import yfinance as yf
    except Exception:
        return hist_map
    if not tickers:
        return hist_map

    latest_any = None
    for s in hist_map.values():
        if s is not None and len(s):
            last = s.index.max()
            latest_any = last if (latest_any is None or last > latest_any) else latest_any

    # If we have a last date, start a bit before; otherwise pull the last 10 days.
    start = (latest_any - pd.Timedelta(days=10)) if latest_any is not None else (pd.Timestamp.utcnow() - pd.Timedelta(days=10))
    start = pd.Timestamp(start).tz_localize(None)

    data = yf.download(
        tickers=tickers,
        start=start.date().isoformat(),
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    def _normalize(dfclose: pd.Series, t: str) -> pd.Series:
        ser = dfclose.dropna().copy()
        if isinstance(ser.index, pd.DatetimeIndex) and ser.index.tz is not None:
            ser.index = ser.index.tz_localize(None)
        ser.name = t
        return ser

    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                recent = _normalize(data[(t, "Close")], t)
            except Exception:
                continue
            base = hist_map.get(t, pd.Series(dtype=float))
            combined = pd.concat([base, recent]).sort_index()
            hist_map[t] = combined[~combined.index.duplicated(keep="last")]
    else:
        # single-ticker fallback
        try:
            t0 = tickers[0]
            recent = _normalize(data["Close"], t0)
            base = hist_map.get(t0, pd.Series(dtype=float))
            combined = pd.concat([base, recent]).sort_index()
            hist_map[t0] = combined[~combined.index.duplicated(keep="last")]
        except Exception:
            pass
    return hist_map

# ---------------------------- Yahoo fallback fetchers ----------------------------
@st.cache_data(show_spinner=False, ttl=7200)
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

# ----------------- Empirical Probability (Robust Method #1) -----------------
def compute_empirical_reversion_prob(
    close: pd.Series,
    ma_window: int,
    H: int,
    eps: float,
    bin_edges: np.ndarray,
    min_samples_per_bin: int = 30
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Robust Method #1:
    - Use z vs. rolling mean/stdev.
    - Try the current bin; if sparse, merge with neighbors progressively (±1...±3).
    - Relax min-samples floor in steps if still sparse.
    - Final fallback: unconditional hit rate across all valid samples.
    Returns (probability %, current_z, coverage %).
    """
    if close is None or len(close) < (ma_window + H + 20):
        return (None, None, None)

    mu = close.rolling(ma_window, min_periods=ma_window).mean()
    sd = close.rolling(ma_window, min_periods=ma_window).std(ddof=0)
    valid = (~mu.isna()) & (sd > 0)
    if valid.sum() < (ma_window + H):
        return (None, None, None)

    z = ((close - mu) / sd)
    z = z.where(np.isfinite(z))
    abs_z = z.abs()
    fwd_min = abs_z.shift(-1).rolling(window=H, min_periods=H).min()
    hit = (fwd_min <= eps).astype(float).where(fwd_min.notna())

    z_vals = z.to_numpy()
    hit_vals = hit.to_numpy()
    mask = np.isfinite(z_vals) & np.isfinite(hit_vals)
    if mask.sum() < 10:
        return (None, float(z.iloc[-1]) if np.isfinite(z.iloc[-1]) else None, None)

    bins = np.digitize(z_vals, bin_edges)
    total = int(mask.sum())

    probs_by_bin: Dict[int, float] = {}
    counts_by_bin: Dict[int, int] = {}
    for b in range(1, len(bin_edges) + 1):
        idx = mask & (bins == b)
        cnt = int(idx.sum())
        if cnt > 0:
            counts_by_bin[b] = cnt
            probs_by_bin[b] = float(np.nanmean(hit_vals[idx]))

    current_z = float(z.iloc[-1]) if np.isfinite(z.iloc[-1]) else None
    if current_z is None:
        return (None, None, None)
    current_bin = int(np.digitize([current_z], bin_edges)[0])

    floors = [min_samples_per_bin, max(20, min_samples_per_bin - 10), 15, 10, 5]
    for floor in floors:
        for radius in range(0, 4):  # merge ±0, ±1, ±2, ±3
            merge_bins = [b for b in range(current_bin - radius, current_bin + radius + 1)
                          if 1 <= b <= len(bin_edges)]
            cnt = sum(counts_by_bin.get(b, 0) for b in merge_bins)
            if cnt >= floor:
                num = 0.0
                for b in merge_bins:
                    if b in probs_by_bin and b in counts_by_bin:
                        num += probs_by_bin[b] * counts_by_bin[b]
                p_use = num / cnt if cnt > 0 else np.nan
                if np.isfinite(p_use):
                    return (100.0 * p_use, current_z, 100.0 * (cnt / max(1, total)))

    # Final fallback: unconditional hit rate across all samples
    base = float(np.nanmean(hit_vals[mask]))
    if np.isfinite(base):
        return (100.0 * base, current_z, 100.0)

    return (None, current_z, None)

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
    hist_period = st.selectbox(
        "Historical period for Yahoo fallback",
        options=SUPPORTED_PERIODS,
        index=SUPPORTED_PERIODS.index(DEFAULT_HIST_PERIOD),
        help="Used only when cache is missing for some tickers."
    )

# ---------------------------- Load Master (Parquet) ----------------------------
try:
    base = _load_master_from_parquet(MASTER_PARQUET)
    st.caption(f"Loaded {len(base)} tickers from **{MASTER_PARQUET}**.")
except Exception as e:
    st.error(f"Failed to load master parquet: {e}")
    st.stop()

if base is None or base.empty:
    st.error("No tickers were loaded from the master parquet.")
    st.stop()

# ---------------------------- Add tickers for this run ----------------------------
st.subheader("Optional: Add tickers for this run")
new_raw = st.text_input(
    "Add ticker(s) (comma or space separated, e.g., `SPY, EFA VEA`)",
    value="",
    help="They’ll be UPPERCASED automatically and only used for this run."
)

if new_raw.strip():
    parts = re.split(r"[,\s]+", new_raw.strip())
    parts = [re.sub(r"[^A-Z0-9\.\-\_]", "", p.upper()) for p in parts if p]
    parts = [p for p in parts if p]  # drop empties after cleanup
    add = sorted(set(parts) - set(base["Ticker"]))
    if add:
        st.success(f"Adding {len(add)} new tickers: {', '.join(add)}")
        extra = pd.DataFrame({
            "Ticker": add,
            "Name": add,                 # minimal metadata for new tickers
            "URL": ["" for _ in add],
            "Brand": ["" for _ in add],
            "Category": ["" for _ in add],
            "Region": ["" for _ in add],
            "Expense Ratio (%)": [np.nan for _ in add],
        })
        base = pd.concat([base, extra], ignore_index=True).drop_duplicates(subset=["Ticker"])
    else:
        st.info("No new unique tickers to add.")

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

# ---------------------------- History via Cache + Tail ----------------------------
# 1) Try cached parquet first
hist_map: Dict[str, pd.Series] = _load_cached_history(CACHE_PARQUET, tickers)

# 2) If cache missing or missing tickers, fetch those from Yahoo (fallback)
missing = [t for t in tickers if t not in hist_map or hist_map[t] is None or hist_map[t].empty]
if missing:
    st.caption(f"Fetching history from Yahoo for {len(missing)} missing tickers (first run or uncached).")
    try:
        fresh = _fetch_history_yahoo(missing, period=hist_period)
    except Exception as e:
        st.warning(f"History Yahoo fallback failed: {e}")
        fresh = {}
    for t in missing:
        ser = fresh.get(t)
        if ser is not None:
            hist_map[t] = ser

# 3) Append only the recent tail (~5 trading days via ~10d calendar window)
hist_map = _append_latest_5d(hist_map, tickers)

# (Optional) write back a trimmed 300-day cache if environment permits writes
try:
    rows = []
    for t, s in hist_map.items():
        if s is not None and len(s):
            df_t = s.reset_index()
            df_t.columns = ["Date", "Close"]
            df_t["Ticker"] = t
            rows.append(df_t)
    if rows:
        new_df = pd.concat(rows, ignore_index=True)
        new_df = new_df.sort_values(["Ticker", "Date"])
        new_df = new_df.groupby("Ticker", as_index=False, group_keys=False).tail(300)  # MA200+H support
        CACHE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_parquet(CACHE_PARQUET, index=False)
except Exception:
    pass  # ignore write errors (e.g., read-only container)

# ----------------- Empirical Probability per ticker -----------------
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

# ----------------- Assemble final DF: base meta + snapshot + probabilities -----------------
df = base.merge(px, on="Ticker", how="left")
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
        f"Statistical Buy Prob uses Method #1 on cached+recent Yahoo history, "
        f"MA={ma_for_signal}, H={H}, ε={eps} (robust). "
        f"If the current bin is sparse, we merge neighbors or fall back to unconditional."
    )

# ---------- Top 10: left list (hyperlinks) + right scatter ----------
st.subheader(f"Top 10 • Potential Mean Reversion (below {ma_for_signal}-day MA)")
if top10.empty:
    st.write(f"No entries currently below their {ma_for_signal}-day MA.")
else:
    left, right = st.columns([1, 1])

    # Left: hyperlinked list
    with left:
        for _, r in top10.iterrows():
            ptxt = "" if pd.isna(r["Stat Buy Prob (%)"]) else f" • Stat Prob: {r['Stat Buy Prob (%)']:.1f}%"
            if pd.notna(r.get("URL")) and str(r.get("URL")).strip():
                ticker_md = f"[{r['Ticker']}]({r['URL']})"
            else:
                ticker_md = r["Ticker"]
            st.markdown(
                f"- **{r['Name']} ({ticker_md})** — {abs(r[selected_pct_col]):.2f}% below {ma_for_signal}d MA{ptxt}"
            )

    # Right: scatter plot of depth vs probability with ticker labels (always tries to render)
    with right:
        import matplotlib.pyplot as plt

        plot_df = top10.copy()
        plot_df["Depth Below MA %"] = plot_df[selected_pct_col].abs()
        plot_df["Prob %"] = plot_df["Stat Buy Prob (%)"]

        # Fallback for any missing probability: unconditional rate so chart can still render
        def _fallback_prob_from_series(s: pd.Series, ma_window: int, H: int, eps: float) -> Optional[float]:
            if s is None or len(s) < (ma_window + H + 20):
                return None
            mu = s.rolling(ma_window, min_periods=ma_window).mean()
            sd = s.rolling(ma_window, min_periods=ma_window).std(ddof=0)
            valid = (~mu.isna()) & (sd > 0)
            if valid.sum() < (ma_window + H):
                return None
            z = ((s - mu) / sd).where(np.isfinite((s - mu) / sd))
            abs_z = z.abs()
            fwd_min = abs_z.shift(-1).rolling(window=H, min_periods=H).min()
            hit = (fwd_min <= eps).astype(float).where(fwd_min.notna())
            base = float(np.nanmean(hit))
            return 100.0 * base if np.isfinite(base) else None

        if plot_df["Prob %"].isna().any():
            fill_vals = []
            for _, r in plot_df.iterrows():
                if pd.isna(r["Prob %"]):
                    s = hist_map.get(r["Ticker"])
                    val = _fallback_prob_from_series(s, int(ma_for_signal), int(H), float(eps))
                    fill_vals.append(val)
                else:
                    fill_vals.append(r["Prob %"])
            plot_df["Prob %"] = fill_vals

        plot_df2 = plot_df[plot_df["Prob %"].notna() & plot_df["Depth Below MA %"].notna()]
        n = len(plot_df2)
        # Dynamic size so all labels fit (esp. when you truly have 10)
        fig_h = max(4.2, 0.45 * max(1, n) + 2.0)
        fig_w = 7.8

        if not plot_df2.empty:
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
            ax.scatter(plot_df2["Depth Below MA %"], plot_df2["Prob %"])
            for _, row in plot_df2.iterrows():
                ax.annotate(
                    row["Ticker"],
                    (row["Depth Below MA %"], row["Prob %"]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=9
                )
            ax.set_xlabel(f"Depth below {ma_for_signal}-day MA (%)")
            ax.set_ylabel(f"Stat Buy Prob (%) (H={H}, ε={eps})")
            ax.set_title("Top 10: Mean-Reversion Setup")
            x_max = float(plot_df2["Depth Below MA %"].max())
            y_max = float(plot_df2["Prob %"].max())
            ax.set_xlim(left=0, right=max(5.0, x_max * 1.20 + 1.0))
            ax.set_ylim(bottom=0, top=max(50.0, y_max * 1.15 + 2.0))
            ax.margins(x=0.08, y=0.12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
        else:
            # last resort (should be rare with 5y cache)
            st.info("Not enough valid points to plot.")

st.divider()
st.subheader("All Instruments • Sort columns, search, and download")

# ---------------------------- Table Data (keep numerics for proper sorting) ----------------------------
display = {
    "Name": df["Name"],
    "Ticker": df["Ticker"],
    "Category": df["Category"],
    "Region": df["Region"],
    "Expense Ratio (%)": df["Expense Ratio (%)"],
    "Expense Ratio (Yahoo %)": df["Expense Ratio (Yahoo %)"],
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
for maybe_empty in ["Expense Ratio (Yahoo %)", "Yield % (Yahoo)"]:
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
for c in ["Price vs 50d %", "Price vs 200d %", "Prem/Disc to NAV %", "Expense Ratio (%)", "Expense Ratio (Yahoo %)", "Yield % (Yahoo)"]:
    if c in display_cols.columns: colcfg[c] = st.column_config.NumberColumn(format="%.2f%%")
if "Net Assets (USD)" in display_cols.columns:
    colcfg["Net Assets (USD)"] = st.column_config.NumberColumn(format="$%,.0f")

# One-line caption showing cache vs. fresh, and that a small tail was appended
st.caption(f"History source • Parquet cache: {len(tickers) - len(missing)} tickers | Yahoo (new/uncached): {len(missing)} | + recent ~5 trading days appended")

st.dataframe(display_cols[final_cols], use_container_width=True, column_config=colcfg)

csv_bytes = display_cols[final_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download table (CSV)", data=csv_bytes, file_name="etf_metrics.csv", mime="text/csv")

st.caption(
    "Empirical probability (Robust Method #1): bucket historical z-scores vs the selected MA; "
    "if the current bin is sparse, we merge neighboring bins and, if needed, fall back to the unconditional hit rate."
)
