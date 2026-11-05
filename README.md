# New-Product Extraction from SEC 8-K Filings (NER + LLM)

Extracts newly announced products from SEC Form 8-K filings of S&P 500 companies and saves them as a structured CSV.
Hybrid pipeline: SEC EDGAR API → press-release exhibits → spaCy + rule-based NER → local LLM (Ollama, JSON schema) → validation.

```
company_name, stock_name, filing_time, new_product, product_description
```

## Pipeline

| Step | What it does |
|---|---|
| Universe | S&P 500 constituents (ticker, name, CIK) from Wikipedia, cached to `data/sp500.csv` |
| Filings | `data.sec.gov/submissions` JSON → 8-Ks in the date window with Item 7.01 or 8.01 |
| Documents | Full submission `.txt` → 8-K body + EX-99.x press releases → clean text (iXBRL/HTML stripped) |
| Pre-filter | Launch-language regex; filings without it never reach the LLM |
| NER | spaCy `en_core_web_sm` ORG/PRODUCT entities + rule-based product pattern; top sentences become the LLM passage, entities become hints |
| LLM | Ollama `/api/chat`, temperature 0, JSON-schema output: `product_name`, `description`, `evidence` |
| Validation | Product name must appear in the filing, generic names dropped, duplicates merged, description < 180 chars |
| Output | `output/new_products.csv` (products only), `output/all_filings.csv` (every filing), `output/extraction_audit.csv`, `output/new_products_table.txt` |

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull llama3.1:8b          # deepseek-r1:8b also works (<think> blocks are stripped)

export SEC_USER_AGENT="Your Name your.email@school.edu"   # required by SEC fair-access policy
python extract_8k_products.py --max-docs 100                # or open 8k_product_extraction.ipynb
```

Options: `--tickers AAPL,NVDA` (subset), `--start/--end YYYY-MM-DD`, `--model deepseek-r1:8b`, `--timestamp` (acceptance time instead of date), `--backend openai --model <model>` (needs `OPENAI_API_KEY`).

Runs are resumable: filings already in `extraction_audit.csv` are skipped, and raw filings are cached in `data/cache/`.

## Tests

```bash
pytest -q        # offline: SEC and the LLM are mocked
```

## Files

- `extract_8k_products.py` – pipeline module and CLI
- `8k_product_extraction.ipynb` – notebook driver with NER demo, run, metrics, manual-review sample
- `make_all_filings_csv.py` – builds `output/all_filings.csv`: one row per filing analysed (`None` when no new product)
- `tests/test_pipeline.py` – offline unit and end-to-end tests
- `output/` – CSV results (committed after the run)
