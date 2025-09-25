# tools/prewarm_data.py
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

def load_master(path: Path, sheet: str) -> pd.DataFrame:
    """
    Load the Excel master and return a normalized dataframe.
    At minimum includes a 'Ticker' column (uppercased, deduped).
    If extra columns exist (Name/URL/etc.), keep them in normalized form.
    """
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    cols_lower = [str(c).strip().lower() for c in df.columns]
    df.columns = cols_lower

    # try to keep common metadata columns if they exist
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

    # fallback: column B as ticker if not explicitly found
    if col_map["ticker"] is None and len(df.columns) > 1:
        col_map["ticker"] = df.columns[1]
    if col_map["ticker"] is None:
        raise ValueError("Ticker column not found; expected a 'Ticker' column or a second column (B).")

    # build normalized output
    out = pd.DataFrame()
    out["Ticker"] = (
        df[col_map["ticker"]]
        .astype(str).str.strip().str.upper()
        .replace({"": np.nan})
    )

    def s_blank(key: str) -> pd.Series:
        col = col_map[key]
        if col is None:
            return pd.Series([""] * len(df), dtype="object")
        return df[col].astype(str).fillna("").str.strip()

    # optional metadata
    out["Name"] = s_blank("fund_name")
    out["URL"] = s_blank("hyperlink")
    out["Brand"] = s_blank("brand")
    out["Category"] = s_blank("focus").str.title()
    out["Region"] = s_blank("region")

    # expense ratio (raw string kept as-is here; app can parse to % later if needed)
    exp_col = col_map["expense_ratio"]
    out["Expense Ratio Raw"] = df[exp_col] if exp_col is not None else np.nan

    # drop blanks/dupes
    out = out.dropna(subset=["Ticker"])
    out = out[out["Ticker"] != ""].drop_duplicates(subset=["Ticker"]).reset_index(drop=True)

    # if Name missing, backfill with Brand then Ticker
    name_blank = (out["Name"] == "") | out["Name"].isna()
    out.loc[name_blank, "Name"] = out.loc[name_blank, "Brand"]
    name_blank = (out["Name"] == "") | out["Name"].isna()
    out.loc[name_blank, "Name"] = out.loc[name_blank, "Ticker"]

    return out

def fetch_history_yahoo(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """
    Fetch OHLCV for the given period from Yahoo and return long (Date, Ticker, Close).
    Example period: '5y', '3y', '2y', '1y', 'max', or '730d', etc.
    """
    import yfinance as yf

    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    rows = []
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                ser = data[(t, "Close")].dropna().copy()
                if isinstance(ser.index, pd.DatetimeIndex) and ser.index.tz is not None:
                    ser.index = ser.index.tz_localize(None)
                df_t = ser.reset_index()
                df_t.columns = ["Date", "Close"]
                df_t["Ticker"] = t
                rows.append(df_t)
            except Exception:
                continue
    else:
        # Single-ticker fallback
        try:
            ser = data["Close"].dropna().copy()
            if isinstance(ser.index, pd.DatetimeIndex) and ser.index.tz is not None:
                ser.index = ser.index.tz_localize(None)
            df_t = ser.reset_index()
            df_t.columns = ["Date", "Close"]
            df_t["Ticker"] = tickers[0] if tickers else "TICKER"
            rows.append(df_t)
        except Exception:
            pass

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Date", "Close", "Ticker"])
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    out = out.sort_values(["Ticker", "Date"], kind="mergesort")  # stable sort
    return out[["Date", "Ticker", "Close"]]

def main():
    ap = argparse.ArgumentParser(description="Pre-warm Yahoo history to Parquet and save master copies.")
    ap.add_argument("--excel", default="data/ETF project.xlsm", help="Path to Excel master.")
    ap.add_argument("--sheet", default="Tickers", help="Sheet name.")
    ap.add_argument("--out", default="data/cache", help="Output directory for parquet + copies.")
    ap.add_argument("--period", default="5y", help="Yahoo period to fetch, e.g. 5y, 3y, 2y, 1y, max, or Nd.")
    ap.add_argument("--days", type=int, default=0, help="If >0, trim to last N trading days after fetch (0=keep all).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_path = Path(args.excel)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel master not found at {excel_path}")

    # Load + normalize master (keeps Ticker + common metadata if present)
    base = load_master(excel_path, args.sheet)
    tickers = base["Ticker"].tolist()
    if not tickers:
        raise ValueError("No tickers found in the Excel master.")

    print(f"Tickers: {len(tickers)}")

    # Fetch full period (e.g., 5y), then optionally trim to last N days
    hist_all = fetch_history_yahoo(tickers, period=args.period)

    if args.days and args.days > 0:
        # Trim to last N trading rows per ticker
        hist = (
            hist_all.sort_values(["Ticker", "Date"])
                    .groupby("Ticker", group_keys=False)
                    .tail(args.days)
        )
        days_kept = args.days
    else:
        hist = hist_all
        days_kept = 0  # 0 signals "kept full period"

    pq_path = out_dir / "price_history.parquet"
    hist.to_parquet(pq_path, index=False)  # requires pyarrow
    print(f"Wrote {len(hist)} rows to {pq_path}")

    # Save normalized master as BOTH Excel and Parquet
    excel_copy = out_dir / "ETF_master_copy.xlsx"
    base.to_excel(excel_copy, index=False)

    master_pq = out_dir / "ETF_master_copy.parquet"
    base.to_parquet(master_pq, index=False)  # fast runtime read path
    print(f"Wrote master to {excel_copy.name} and {master_pq.name}")

    # Keep a binary copy of the original Excel (optional)
    orig_copy = out_dir / excel_path.name
    if excel_path.resolve() != orig_copy.resolve():
        try:
            orig_copy.write_bytes(excel_path.read_bytes())
        except Exception:
            pass

    # Metadata
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "period_fetched": args.period,
        "days_kept": days_kept,
        "excel_source": str(excel_path),
        "sheet": args.sheet,
        "rows": int(len(hist)),
        "tickers": int(len(tickers)),
        "artifacts": {
            "price_history_parquet": str(pq_path),
            "master_excel": str(excel_copy),
            "master_parquet": str(master_pq),
            "orig_excel_copy": str(orig_copy),
        },
    }
    (out_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.")

if __name__ == "__main__":
    main()
