
from __future__ import annotations
import pandas as pd
from pathlib import Path

def _choose_sheet(xls: pd.ExcelFile) -> str:
    # Prefer a sheet that contains 'Data Structure 2' in the name
    for s in xls.sheet_names:
        if "Data Structure 2" in s:
            return s
    return xls.sheet_names[0]

def load_ecosystem_data(excel_path: str | Path) -> pd.DataFrame:
    """
    Load the 'Data Structure 2' sheet from the provided Excel, normalizing columns:
    - Sector        -> 'Sector'
    - sub-sector    -> 'Subsector'
    - FINTECH       -> 'Name'
    - Hyperlinks    -> 'URL'
    - Description   -> 'Description' (optional; handles 'Dezcription / Key words')
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    sheet = _choose_sheet(xls)

    # Try header at row 2 (skip the first two rows). Fallbacks included.
    try_orders = [2, 1, 0, None]
    df = None
    for hdr in try_orders:
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet, header=hdr, engine="openpyxl")
            # Normalize column names for inspection
            df.columns = [str(c).strip() for c in df.columns]
            if "Sector" in df.columns and "sub-sector" in df.columns:
                break
        except Exception:
            df = None
            continue

    if df is None:
        # Last resort: read default and hope headers are correct
        df = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]

    # Build column map
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "sector":
            col_map[c] = "Sector"
        elif cl in ("sub-sector", "subsector", "sub-sector "):
            col_map[c] = "Subsector"
        elif cl == "fintech":
            col_map[c] = "Name"
        elif cl.replace(" ", "") in ("hyperlinks", "hyperlink", "url"):
            col_map[c] = "URL"
        elif "description" in cl or "dezcription" in cl:
            col_map[c] = "Description"

    df = df.rename(columns=col_map)

    required = {"Sector", "Subsector", "Name"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found in '{sheet}': {missing}")

    # If URL or Description are missing, create empty columns
    if "URL" not in df.columns:
        df["URL"] = ""
    if "Description" not in df.columns:
        df["Description"] = ""

    # Clean up
    for col in ["Sector", "Subsector", "Name", "URL", "Description"]:
        df[col] = df[col].astype(str).str.strip()

    # Drop completely empty name rows
    df = df[df["Name"] != ""]
    df = df.dropna(subset=["Sector", "Subsector", "Name"])

    # Keep only normalized columns (plus any others if needed later)
    return df[["Sector", "Subsector", "Name", "URL", "Description"]]
