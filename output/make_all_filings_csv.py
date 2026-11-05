"""
make_all_filings_csv.py
=======================

Build output/all_filings.csv: one row per 8-K filing the pipeline analysed, in the
assignment's five columns (company_name, stock_name, filing_time, new_product,
product_description).

- Filings where the LLM found new products repeat their rows from new_products.csv.
- Filings without a product get new_product = "None" and a description saying why:
  the keyword pre-filter found no launch language, or the LLM found no new product.

new_products.csv stays the products-only result; this file shows every document analysed.

Usage (after extract_8k_products.py has run or been stopped):
    python make_all_filings_csv.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

import extract_8k_products as ex

NO_PRODUCT_REASON = {
    "skipped_no_launch_language": "No product-launch language in this filing (keyword pre-filter).",
    "llm_no_product": "LLM found no new product in this filing.",
}


def build_all_filings(out_dir: Path = Path("output"), sp500_csv: Path = Path("data/sp500.csv")) -> pd.DataFrame:
    """Combine the audit log (every filing) with the product rows into one table."""
    audit = pd.read_csv(out_dir / "extraction_audit.csv", dtype=str).fillna("")
    products_path = out_dir / "new_products.csv"
    products = (pd.read_csv(products_path, dtype=str).fillna("") if products_path.exists()
                else pd.DataFrame(columns=ex.OUTPUT_COLUMNS))

    # ticker -> company name, from the same S&P 500 list the pipeline used
    universe = ex.load_sp500(ex.Config(user_agent="offline offline@example.com", sp500_csv=sp500_csv))
    company = dict(zip(universe["ticker"], universe["company"]))

    rows = []
    for a in audit.itertuples(index=False):
        name = company.get(a.ticker, a.ticker)
        if a.status == "llm_products":
            for product in [p for p in a.products.split("; ") if p]:
                match = products[(products["stock_name"] == a.ticker)
                                 & (products["filing_time"].str[:10] == a.filing_date)
                                 & (products["new_product"] == product)]
                desc = match["product_description"].iloc[0] if len(match) else ""
                ftime = match["filing_time"].iloc[0] if len(match) else a.filing_date
                rows.append([name, a.ticker, ftime, product, desc])
        else:
            reason = NO_PRODUCT_REASON.get(a.status, f"No product ({a.status}).")
            rows.append([name, a.ticker, a.filing_date, "None", reason])
    return pd.DataFrame(rows, columns=ex.OUTPUT_COLUMNS)


if __name__ == "__main__":
    out = Path("output")
    df = build_all_filings(out)
    df.to_csv(out / "all_filings.csv", index=False, quoting=csv.QUOTE_ALL)
    n_prod = (df["new_product"] != "None").sum()
    print(f"Wrote {out / 'all_filings.csv'}: {len(df)} rows "
          f"({n_prod} product rows, {len(df) - n_prod} filings without a new product, "
          f"{df['stock_name'].nunique()} companies).")
