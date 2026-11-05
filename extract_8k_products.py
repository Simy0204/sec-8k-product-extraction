"""
extract_8k_products.py
======================

Extract new-product announcements from SEC Form 8-K filings with a hybrid
NER + LLM pipeline, and save them as a CSV:

    company_name, stock_name, filing_time, new_product, product_description

Pipeline
--------
1. Universe      S&P 500 constituents (ticker, name, CIK)            -> Wikipedia / local CSV
2. Filings       8-Ks per company from the SEC submissions JSON API   -> data.sec.gov
                 (kept: Item 7.01 Reg FD / Item 8.01 Other Events, where press releases live)
3. Documents     full submission .txt -> 8-K body + EX-99.x press releases -> clean text
4. Pre-filter    launch-keyword regex (skip filings with no product language, saves LLM calls)
5. NER           spaCy en_core_web_sm: ORG / PRODUCT entities + rule-based product patterns;
                 used to pick the most relevant sentences and passed to the LLM as hints
6. LLM           local model via Ollama (default llama3.1:8b; DeepSeek works too) with a JSON
                 schema -> list of {product_name, evidence, announcement_type, offered_by_filer,
                 description}
7. Validation    product name must appear in the filing text (anti-hallucination), only
                 new_product types offered by the filer, generic names dropped, duplicates
                 merged, description < 180 characters
7b. Verification second LLM pass: a focused yes/no question per surviving candidate, using the
                 press-release opening + the sentences that mention it (drops deals, acquisitions,
                 facilities and other companies' products that the first pass let through)
8. Output        new_products.csv (assignment format), extraction_audit.csv (every filing,
                 its source URL and outcome), new_products_table.txt (pipe-separated table)

Usage
-----
    ollama pull llama3.1:8b
    export SEC_USER_AGENT="Your Name your.email@school.edu"
    python extract_8k_products.py --max-docs 100

SEC fair-access rules: identify yourself in the User-Agent and stay under 10 requests/second.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

OUTPUT_COLUMNS = ["company_name", "stock_name", "filing_time", "new_product", "product_description"]
AUDIT_COLUMNS = [
    "accession", "cik", "ticker", "filing_date", "acceptance_time", "items", "url",
    "status", "n_products", "products", "rejected", "ner_orgs", "ner_products",
    "verify_notes", "llm_raw",
]
MAX_DESC_CHARS = 179  # "less than 180 characters"


@dataclass
class Config:
    user_agent: str = os.environ.get("SEC_USER_AGENT", "")
    start_date: str = (date.today() - timedelta(days=365)).isoformat()
    end_date: str = date.today().isoformat()
    max_documents: int = 100            # filings sent to the LLM (after the keyword pre-filter)
    max_filings_per_company: int = 15   # newest-first, keeps the sample spread across companies
    items_filter: tuple = ("7.01", "8.01")
    tickers: list | None = None         # optional subset, e.g. ["AAPL", "NVDA"]
    backend: str = "ollama"             # "ollama" or "openai" (any OpenAI-compatible endpoint)
    model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    openai_url: str = "https://api.openai.com/v1"
    passage_chars: int = 6000           # text budget sent to the LLM per filing
    filing_time_format: str = "date"    # "date" -> 2025-02-19, "timestamp" -> 2025-02-19 16:05:12
    verify: bool = True                 # second LLM pass: yes/no check per candidate product
    out_dir: Path = Path("output")
    cache_dir: Path = Path("data/cache")
    sp500_csv: Path = Path("data/sp500.csv")  # optional local constituents file
    sec_sleep: float = 0.15

    @property
    def out_csv(self) -> Path:
        return self.out_dir / "new_products.csv"

    @property
    def audit_csv(self) -> Path:
        return self.out_dir / "extraction_audit.csv"

    @property
    def table_txt(self) -> Path:
        return self.out_dir / "new_products_table.txt"


# --------------------------------------------------------------------------------------
# 1-3. SEC EDGAR access
# --------------------------------------------------------------------------------------

class SecClient:
    """Polite EDGAR client: identified User-Agent, rate limit, retries, disk cache."""

    def __init__(self, cfg: Config):
        if not cfg.user_agent or "@" not in cfg.user_agent:
            raise ValueError(
                'SEC requires a User-Agent with your name and email, e.g. '
                'export SEC_USER_AGENT="Jane Doe jane@school.edu"'
            )
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent, "Accept-Encoding": "gzip, deflate"})
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, url: str, cache_name: str | None = None) -> str:
        if cache_name:
            path = self.cfg.cache_dir / cache_name
            if path.exists():
                return path.read_text(encoding="utf-8", errors="ignore")
        for attempt in range(5):
            time.sleep(self.cfg.sec_sleep)
            r = self.session.get(url, timeout=30)
            if r.status_code == 200:
                if cache_name:
                    (self.cfg.cache_dir / cache_name).write_text(r.text, encoding="utf-8")
                return r.text
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        raise RuntimeError(f"SEC request kept failing: {url}")


def load_sp500(cfg: Config) -> pd.DataFrame:
    """S&P 500 constituents as columns ticker, company, cik (int)."""
    if cfg.sp500_csv.exists():
        df = pd.read_csv(cfg.sp500_csv)
    else:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (course project)"}, timeout=30).text
        df = pd.read_html(io.StringIO(html))[0]
        cfg.sp500_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.sp500_csv, index=False)
    df = df.rename(columns={"Symbol": "ticker", "Security": "company", "CIK": "cik"})
    df = df[["ticker", "company", "cik"]].dropna()
    df["cik"] = df["cik"].astype(int)
    if cfg.tickers:
        wanted = {t.upper() for t in cfg.tickers}
        df = df[df["ticker"].str.upper().isin(wanted)]
    return df.sort_values("ticker").reset_index(drop=True)


def list_8k_filings(sec: SecClient, cik: int, cfg: Config) -> list[dict]:
    """8-K filings in the date window whose Items include 7.01 or 8.01, newest first."""
    raw = sec.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    sub = json.loads(raw)
    recent = sub.get("filings", {}).get("recent", {})
    out = []
    for i, form in enumerate(recent.get("form", [])):
        if form != "8-K":
            continue
        fdate = recent["filingDate"][i]
        if not (cfg.start_date <= fdate <= cfg.end_date):
            continue
        items = recent.get("items", [""] * (i + 1))[i] or ""
        if cfg.items_filter and not any(it in items.split(",") for it in cfg.items_filter):
            continue
        out.append({
            "accession": recent["accessionNumber"][i],
            "filing_date": fdate,
            "acceptance_time": recent.get("acceptanceDateTime", [""] * (i + 1))[i],
            "items": items,
            "company_edgar": sub.get("name", ""),
        })
    out.sort(key=lambda f: f["filing_date"], reverse=True)
    return out[: cfg.max_filings_per_company]


def filing_url(cik: int, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{accession}.txt"


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def split_submission(raw: str) -> list[dict]:
    """Split a full EDGAR submission (.txt) into its documents; keep the 8-K body and EX-99.x."""
    docs = []
    for block in re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", raw, flags=re.S | re.I):
        m_type = re.search(r"<TYPE>([^\n<]+)", block, flags=re.I)
        doc_type = m_type.group(1).strip().upper() if m_type else ""
        if not (doc_type == "8-K" or doc_type.startswith("EX-99")):
            continue
        m_text = re.search(r"<TEXT>(.*?)</TEXT>", block, flags=re.S | re.I)
        body = m_text.group(1) if m_text else block
        m_name = re.search(r"<FILENAME>([^\n<]+)", block, flags=re.I)
        text = html_to_text(body) if re.search(r"<(html|div|p|table)\b", body, re.I) else re.sub(r"\s+", " ", body)
        docs.append({"type": doc_type, "filename": m_name.group(1).strip() if m_name else "", "text": text})
    return docs


def build_filing_text(docs: list[dict], max_chars: int = 60000) -> str:
    """Press releases (EX-99.x) first, then the 8-K cover document."""
    ordered = sorted(docs, key=lambda d: (0 if d["type"].startswith("EX-99") else 1, d["type"]))
    return " \n".join(d["text"] for d in ordered)[:max_chars]


# --------------------------------------------------------------------------------------
# 4-5. Keyword pre-filter and Named Entity Recognition
# --------------------------------------------------------------------------------------

LAUNCH_RE = re.compile(
    r"\b(launch(?:es|ed|ing)?|introduc(?:e|es|ed|ing)|unveil(?:s|ed|ing)?|debut(?:s|ed)?|"
    r"roll(?:s|ed)? out|now available|generally available|general availability|"
    r"new (?:products?|platforms?|services?|solutions?|models?|features?|line)|"
    r"announc(?:e|es|ed|ing) (?:the )?(?:launch|availability|release|new)|"
    r"FDA approv(?:al|ed|es))\b",
    re.I,
)

# Rule-based NER: a capitalised phrase right after a launch verb ("... launched the Pixel 10 Pro ...")
PRODUCT_PATTERN = re.compile(
    r"(?i:launch(?:es|ed)?|introduc(?:es|ed)|unveil(?:s|ed)|debut(?:s|ed)|"
    r"announc(?:es|ed) (?:the )?(?:launch|availability|general availability) of)"
    r"\s+(?:of\s+)?(?:(?i:the|its|their|a|an)\s+)?"
    r"(?:(?i:new|all-new|latest|newest|next-generation|next-gen)\s+){0,2}"
    r"([A-Z0-9][\w\-.®™+/]*(?:\s+[A-Z0-9][\w\-.®™+/]*){0,5})"
)
_TRAILING = {"The", "A", "An", "In", "On", "At", "For", "And", "To", "Of", "With", "Today", "This"}


def load_nlp():
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except OSError as e:
        raise OSError("Run: python -m spacy download en_core_web_sm") from e
    nlp.max_length = 200_000
    return nlp


def rule_based_products(text: str) -> list[str]:
    found = []
    for m in PRODUCT_PATTERN.finditer(text):
        words = m.group(1).split()
        while words and (words[-1] in _TRAILING or not words[-1][0].isalnum()):
            words.pop()
        name = " ".join(words).strip(" .,;:")
        if len(name) >= 2:
            found.append(name)
    return found


def run_ner(nlp, text: str, passage_chars: int = 6000) -> tuple[str, dict]:
    """
    Run spaCy NER, score each sentence for product relevance and return
    (passage for the LLM, entity candidates).
    """
    doc = nlp(text[:100_000])
    orgs, products = Counter(), Counter()
    scored = []
    for idx, sent in enumerate(doc.sents):
        s_text = sent.text.strip()
        if len(s_text) < 20:
            continue
        ents = [e for e in sent.ents]
        for e in ents:
            if e.label_ == "ORG":
                orgs[e.text.strip()] += 1
            elif e.label_ == "PRODUCT":
                products[e.text.strip()] += 1
        score = 0
        if LAUNCH_RE.search(s_text):
            score += 3
        score += 2 * sum(e.label_ == "PRODUCT" for e in ents)
        score += min(2, sum(e.label_ == "ORG" for e in ents))
        if idx < 3:  # headline + dateline of the press release
            score += 2
        scored.append((idx, score, s_text))

    for name in rule_based_products(text):
        products[name] += 1

    # keep the best sentences, then restore original order so the passage reads naturally
    keep, used = [], 0
    for idx, score, s_text in sorted(scored, key=lambda t: (-t[1], t[0])):
        if score <= 0 or used + len(s_text) > passage_chars:
            continue
        keep.append((idx, s_text))
        used += len(s_text) + 1
    passage = " ".join(s for _, s in sorted(keep)) or text[:passage_chars]

    candidates = {
        "ORG": [n for n, _ in orgs.most_common(8)],
        "PRODUCT": [n for n, _ in products.most_common(10)],
    }
    return passage, candidates


# --------------------------------------------------------------------------------------
# 6. LLM extraction
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a financial information-extraction system that performs named entity "
    "recognition on SEC Form 8-K filings. You extract NEW products or services that the "
    "filing company announces, launches, introduces, or makes available. You answer with JSON only."
)

USER_TEMPLATE = """Filing company: {company} (ticker {ticker}), 8-K filed {filing_date}.

NER hints from spaCy (may be incomplete or wrong):
- ORGANIZATION entities: {orgs}
- PRODUCT candidates: {products}

Rules:
1. A product is a specific, named offering: hardware, software, AI model, drug or device, vehicle,
   financial product, service or platform that the FILING COMPANY itself sells or offers and is
   launching, releasing or making available for the first time (a new version or model counts).
2. NOT products: earnings, guidance, dividends, buybacks, debt offerings, acquisitions or investments,
   partnerships and customer or supply deals, data centers, campuses and other facilities, executive
   changes, lawsuits, another company's product, or products only mentioned as already existing.
3. product_name: the name exactly as written in the text (max 8 words).
4. evidence: a short verbatim quote (max 25 words) from the text that shows the announcement.
5. announcement_type: classify the evidence honestly, using exactly one of:
   new_product, partnership_or_deal, investment_or_acquisition, facility_or_infrastructure,
   financial_or_corporate, other.
6. offered_by_filer: true only if {company} itself sells or offers the product.
7. description: one factual sentence, under 180 characters, using only facts from the text.
8. If there is no new product, return {{"products": []}}. Several products -> several entries.

Return JSON: {{"products": [{{"product_name": "...", "evidence": "...", "announcement_type": "new_product",
"offered_by_filer": true, "description": "..."}}]}}

Filing text:
\"\"\"{passage}\"\"\"
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                # order matters: the model quotes evidence, then classifies, then describes
                "properties": {
                    "product_name": {"type": "string"},
                    "evidence": {"type": "string"},
                    "announcement_type": {"type": "string", "enum": [
                        "new_product", "partnership_or_deal", "investment_or_acquisition",
                        "facility_or_infrastructure", "financial_or_corporate", "other"]},
                    "offered_by_filer": {"type": "boolean"},
                    "description": {"type": "string"},
                },
                "required": ["product_name", "evidence", "announcement_type", "offered_by_filer",
                             "description"],
            },
        }
    },
    "required": ["products"],
}


def build_prompt(company: str, ticker: str, filing_date: str, passage: str, candidates: dict) -> str:
    return USER_TEMPLATE.format(
        company=company, ticker=ticker, filing_date=filing_date, passage=passage,
        orgs=", ".join(candidates.get("ORG", [])) or "none",
        products=", ".join(candidates.get("PRODUCT", [])) or "none",
    )


VERIFY_TEMPLATE = """Company: {company}
Candidate product: "{name}"

Text from {company}'s SEC 8-K filing:
\"\"\"{context}\"\"\"

Question: In this text, is "{name}" itself a product or service that {company} sells or offers and
is launching, releasing or making available for the first time?

Answer "no" if the candidate is not itself being launched by {company}, for example if it is
- a company, business or platform being acquired or invested in,
- the subject of a partnership, customer, supply or hosting deal,
- a facility, campus, data center or site,
- a product made or owned by another company,
- an existing product that is only mentioned.

First give a short reason (max 20 words), then the answer.
Return JSON: {{"reason": "...", "answer": "yes" or "no"}}
"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "answer": {"type": "string", "enum": ["yes", "no"]},
    },
    "required": ["reason", "answer"],
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def verify_context(name: str, text: str, max_chars: int = 2000) -> str:
    """Opening of the press release + every sentence that mentions the candidate (with neighbours)."""
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sents:
        return text[:max_chars]
    hits = [i for i, s in enumerate(sents) if is_grounded(name, s)]
    keep = sorted({j for i in hits for j in (i - 1, i, i + 1) if 0 <= j < len(sents)})
    opening = " ".join(sents[:2])[:400]
    body, used = [], len(opening)
    for j in keep:
        if j < 2:
            continue
        if used + len(sents[j]) > max_chars:
            break
        body.append(sents[j])
        used += len(sents[j]) + 1
    return (opening + " ... " + " ".join(body)).strip(" .") if body else opening


def parse_verify(raw: str) -> tuple[bool | None, str]:
    """(True/False, reason); None when the answer cannot be parsed."""
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    start, end = raw.find("{"), raw.rfind("}")
    try:
        data = json.loads(raw[start:end + 1]) if start != -1 and end > start else {}
    except json.JSONDecodeError:
        data = {}
    ans = str(data.get("answer", "")).strip().lower()
    reason = re.sub(r"\s+", " ", str(data.get("reason", ""))).strip()[:200]
    if ans.startswith("y"):
        return True, reason
    if ans.startswith("n"):
        return False, reason
    return None, reason


def verify_products(valid: list[dict], text: str, company: str,
                    llm: Callable[..., str]) -> tuple[list[dict], list[str], list[str]]:
    """Second LLM pass: a focused yes/no check for each candidate. Returns (kept, rejected, notes)."""
    kept, rejected, notes = [], [], []
    for v in valid:
        prompt = VERIFY_TEMPLATE.format(company=company, name=v["new_product"],
                                        context=verify_context(v["new_product"], text))
        ok, reason = parse_verify(llm(prompt, schema=VERIFY_SCHEMA))
        notes.append(f"{v['new_product']}: {'yes' if ok else 'no' if ok is False else 'unparsed'} ({reason})")
        if ok is False:
            rejected.append(f"{v['new_product']}: verifier said no ({reason})")
        else:  # yes, or unparsable answer -> keep (fail open, noted in the audit)
            kept.append(v)
    return kept, rejected, notes


def make_llm(cfg: Config) -> Callable[..., str]:
    """Return a function (prompt, schema=...) -> raw model text."""
    if cfg.backend == "ollama":
        def call(prompt: str, schema: dict = JSON_SCHEMA) -> str:
            payload = {
                "model": cfg.model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": prompt}],
                "stream": False,
                "format": schema,  # structured output (Ollama >= 0.5); older versions: "json"
                "options": {"temperature": 0, "num_ctx": 8192},
            }
            r = requests.post(f"{cfg.ollama_url}/api/chat", json=payload, timeout=600)
            r.raise_for_status()
            return r.json()["message"]["content"]
        return call

    if cfg.backend == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        def call(prompt: str, schema: dict = JSON_SCHEMA) -> str:  # schema enforced via the prompt
            payload = {
                "model": cfg.model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            r = requests.post(f"{cfg.openai_url}/chat/completions", json=payload,
                              headers={"Authorization": f"Bearer {key}"}, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        return call

    raise ValueError(f"Unknown backend: {cfg.backend}")


def parse_llm_json(raw: str) -> list[dict]:
    """Tolerant JSON parsing: strips DeepSeek <think> blocks and markdown fences."""
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S)
    raw = re.sub(r"```(?:json)?", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    items = data.get("products", []) if isinstance(data, dict) else []
    return [i for i in items if isinstance(i, dict)]


# --------------------------------------------------------------------------------------
# 7. Validation
# --------------------------------------------------------------------------------------

# verbs only, so a product type such as "data acquisition system" is not caught
DEAL_RE = re.compile(r"\b(acquir(?:e|es|ed|ing)|merg(?:e|es|ed|ing) with|merger)\b", re.I)

GENERIC_NAMES = {
    "new product", "new products", "product", "products", "platform", "service", "services",
    "solution", "solutions", "the company", "company", "n/a", "none", "unknown",
}


def _norm(s: str) -> str:
    s = re.sub(r"[®™©]", "", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def is_grounded(name: str, text: str) -> bool:
    """The product name (or >= 80% of its significant tokens) must occur in the filing."""
    n, t = _norm(name), _norm(text)
    if not n:
        return False
    if f" {n} " in f" {t} ":
        return True
    tokens = [w for w in n.split() if len(w) > 2 or w.isdigit()]
    if not tokens:
        return False
    text_tokens = set(t.split())
    return sum(w in text_tokens for w in tokens) / len(tokens) >= 0.8


def clean_description(desc: str, limit: int = MAX_DESC_CHARS) -> str:
    desc = re.sub(r"\s+", " ", (desc or "")).strip().strip('"').strip()
    if len(desc) <= limit:
        return desc
    cut = desc[: limit - 3].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return cut + "..."


def validate(products: list[dict], source_text: str, company: str) -> tuple[list[dict], list[str]]:
    valid, rejected, seen = [], [], set()
    company_n = _norm(company)
    for p in products:
        name = re.sub(r"\s+", " ", str(p.get("product_name", ""))).strip().strip('"')
        key = _norm(name)
        if not key or key in GENERIC_NAMES or key == company_n or len(name) > 100:
            rejected.append(f"{name or '<empty>'}: generic/empty")
            continue
        if not is_grounded(name, source_text):
            rejected.append(f"{name}: not found in filing text")
            continue
        own_words = f"{p.get('description', '')} {p.get('evidence', '')}"
        if DEAL_RE.search(own_words):
            rejected.append(f"{name}: model's own description/evidence describes an acquisition")
            continue
        a_type = str(p.get("announcement_type", "new_product")).strip().lower()
        if a_type != "new_product":
            rejected.append(f"{name}: classified as {a_type}")
            continue
        if p.get("offered_by_filer") is False or str(p.get("offered_by_filer")).lower() == "false":
            rejected.append(f"{name}: not offered by the filing company")
            continue
        if key in seen:
            continue
        seen.add(key)
        desc = clean_description(str(p.get("description", "")))
        if not desc:
            rejected.append(f"{name}: empty description")
            continue
        valid.append({"new_product": name, "product_description": desc,
                      "evidence": str(p.get("evidence", ""))[:300]})
    return valid, rejected


# --------------------------------------------------------------------------------------
# 8. Output
# --------------------------------------------------------------------------------------

def append_rows(path: Path, columns: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        if new:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def done_accessions(audit_csv: Path) -> set[str]:
    if not audit_csv.exists():
        return set()
    return set(pd.read_csv(audit_csv, dtype=str)["accession"].dropna())


def write_pipe_table(out_csv: Path, table_txt: Path) -> str:
    """Pipe-separated table in the assignment's display format."""
    if not out_csv.exists():
        return ""
    df = pd.read_csv(out_csv, dtype=str).fillna("")
    headers = ["Company Name", "Stock Name", "Filing Time", "New Product", "Product Description"]
    widths = [max(len(h), *(len(v) for v in df[c])) if len(df) else len(h)
              for h, c in zip(headers, OUTPUT_COLUMNS)]
    widths[-1] = len(headers[-1])  # don't pad the last column
    line = lambda vals: " | ".join(str(v).ljust(w) for v, w in zip(vals, widths)).rstrip()
    out = [line(headers), "-" * min(140, sum(widths) + 3 * 4 + 60)]
    out += [line(r) for r in df[OUTPUT_COLUMNS].itertuples(index=False)]
    text = "\n".join(out) + "\n"
    table_txt.write_text(text, encoding="utf-8")
    return text


def format_filing_time(f: dict, fmt: str) -> str:
    if fmt == "timestamp" and f.get("acceptance_time"):
        return f["acceptance_time"].replace("T", " ").replace(".000Z", "").rstrip("Z")
    return f["filing_date"]


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def process_filing(f: dict, cik: int, ticker: str, company: str, sec: SecClient, nlp,
                   llm: Callable[..., str], cfg: Config) -> tuple[list[dict], dict]:
    """Run one filing through steps 3-7. Returns (output rows, audit row)."""
    url = filing_url(cik, f["accession"])
    audit = {"accession": f["accession"], "cik": cik, "ticker": ticker, "filing_date": f["filing_date"],
             "acceptance_time": f.get("acceptance_time", ""), "items": f["items"], "url": url,
             "status": "", "n_products": 0, "products": "", "rejected": "", "ner_orgs": "", "ner_products": "",
             "llm_raw": "", "verify_notes": ""}

    raw = sec.get(url, cache_name=f"{f['accession']}.txt")
    text = build_filing_text(split_submission(raw))
    if not LAUNCH_RE.search(text):
        audit["status"] = "skipped_no_launch_language"
        return [], audit

    passage, candidates = run_ner(nlp, text, cfg.passage_chars)
    audit["ner_orgs"] = "; ".join(candidates["ORG"])
    audit["ner_products"] = "; ".join(candidates["PRODUCT"])

    prompt = build_prompt(company, ticker, f["filing_date"], passage, candidates)
    raw_answer = llm(prompt, schema=JSON_SCHEMA)
    audit["llm_raw"] = re.sub(r"\s+", " ", raw_answer or "")[:1500]
    products = parse_llm_json(raw_answer)
    valid, rejected = validate(products, text, company)
    if cfg.verify and valid:
        valid, v_rejected, notes = verify_products(valid, text, company, llm)
        rejected += v_rejected
        audit["verify_notes"] = "; ".join(notes)

    audit.update(status="llm_products" if valid else "llm_no_product", n_products=len(valid),
                 products="; ".join(v["new_product"] for v in valid), rejected="; ".join(rejected))
    rows = [{"company_name": company, "stock_name": ticker,
             "filing_time": format_filing_time(f, cfg.filing_time_format), **v} for v in valid]
    return rows, audit


def run(cfg: Config) -> pd.DataFrame:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    sec = SecClient(cfg)
    nlp = load_nlp()
    llm = make_llm(cfg)
    universe = load_sp500(cfg)
    done = done_accessions(cfg.audit_csv)
    llm_calls, n_products = 0, 0
    print(f"Universe: {len(universe)} companies | window {cfg.start_date}..{cfg.end_date} | "
          f"model {cfg.backend}:{cfg.model} | target {cfg.max_documents} LLM documents")

    for comp in universe.itertuples(index=False):
        try:
            filings = list_8k_filings(sec, int(comp.cik), cfg)
        except Exception as e:  # one bad company must not stop the run
            print(f"  ! {comp.ticker}: could not list filings ({e})")
            continue
        for f in filings:
            if f["accession"] in done:
                continue
            try:
                rows, audit = process_filing(f, int(comp.cik), comp.ticker, comp.company, sec, nlp, llm, cfg)
            except Exception as e:
                print(f"  ! {comp.ticker} {f['accession']}: {e}")
                continue
            append_rows(cfg.out_csv, OUTPUT_COLUMNS, rows)
            append_rows(cfg.audit_csv, AUDIT_COLUMNS, [audit])
            if audit["status"].startswith("llm"):
                llm_calls += 1
                n_products += len(rows)
                print(f"  [{llm_calls:>3}/{cfg.max_documents}] {comp.ticker:<6} {f['filing_date']} "
                      f"-> {audit['products'] or 'no new product'}")
            if llm_calls >= cfg.max_documents:
                print(f"Done: {llm_calls} documents sent to the LLM, {n_products} product rows.")
                write_pipe_table(cfg.out_csv, cfg.table_txt)
                return pd.read_csv(cfg.out_csv) if cfg.out_csv.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)

    print(f"Universe exhausted: {llm_calls} documents sent to the LLM, {n_products} product rows.")
    write_pipe_table(cfg.out_csv, cfg.table_txt)
    return pd.read_csv(cfg.out_csv) if cfg.out_csv.exists() else pd.DataFrame(columns=OUTPUT_COLUMNS)


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Extract new products from SEC 8-K filings with NER + LLM.")
    p.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", ""),
                   help='"Your Name your@email" (or set SEC_USER_AGENT)')
    p.add_argument("--start", default=Config.start_date)
    p.add_argument("--end", default=Config.end_date)
    p.add_argument("--max-docs", type=int, default=100)
    p.add_argument("--tickers", default="", help="comma-separated subset, e.g. AAPL,NVDA,TSLA")
    p.add_argument("--backend", default="ollama", choices=["ollama", "openai"])
    p.add_argument("--model", default="llama3.1:8b")
    p.add_argument("--timestamp", action="store_true", help="filing_time as acceptance timestamp")
    p.add_argument("--out-dir", default="output")
    p.add_argument("--no-verify", action="store_true", help="skip the second-pass yes/no check")
    a = p.parse_args(argv)
    return Config(user_agent=a.user_agent, start_date=a.start, end_date=a.end, max_documents=a.max_docs,
                  tickers=[t.strip() for t in a.tickers.split(",") if t.strip()] or None,
                  backend=a.backend, model=a.model, out_dir=Path(a.out_dir),
                  filing_time_format="timestamp" if a.timestamp else "date", verify=not a.no_verify)


if __name__ == "__main__":
    try:
        df = run(parse_args())
        print(df.head(20).to_string(index=False))
    except (ValueError, OSError) as e:
        sys.exit(f"Error: {e}")
