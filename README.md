
# Fintech Streamlit App

A clean, modular Streamlit app with:

1. **Company Search** – Quick lookup across the ecosystem data by name/keywords.
2. **ETF Mean Reversion** – Yahoo Finance-based snapshot with sortable table and a top‑10 “below MA” placeholder.
3. **Fund Manager Search** – Placeholder for a RAG/Vector search over your JSON/CSV content.
4. **Margin Requirements** -  analyzing margin requirements

---

## Quickstart

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run streamlit_app.py
```

The app auto-discovers pages from the `pages/` folder. Use the sidebar to navigate.

---

## Data Files

- **Fintech Search V15.xlsm**  
  Put this file in `data/` (preferred) or in the project root. The *Ecosystem Map* and *Company Search* pages load:
  - `Sector` (column A)
  - `sub-sector` (column B)
  - `FINTECH` (company name)
  - `Hyperlinks` (column L)
  - `Dezcription / Key words` (optional)

- **config/etfs.txt**  
  The exact list of instruments you provided. You can edit this file to add/remove entries.

---

## Notes

- The **ETF page** uses Yahoo Finance (`yfinance`). Some OTC or non‑ETF tickers may not have complete data; they’ll still appear in the table with blank metrics.
- The **Top‑10** section is a placeholder: it simply surfaces items trading most below their 200‑day moving average. Replace this with your preferred statistical model later.
- Code is intentionally straightforward and well‑commented so it’s easy to extend.
