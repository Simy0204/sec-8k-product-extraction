"""Offline tests: no SEC or LLM access needed (both are mocked). Run with: pytest -q"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_8k_products as ex  # noqa: E402

SAMPLE_SUBMISSION = """<SEC-DOCUMENT>0000000001-25-000001.txt
<DOCUMENT>
<TYPE>8-K
<SEQUENCE>1
<FILENAME>form8k.htm
<TEXT>
<html><body><div style="display:none"><ix:header>hidden xbrl</ix:header></div>
<p>Item 7.01 Regulation FD Disclosure. On February 19, 2025, Example Corp issued a press release,
attached as Exhibit 99.1.</p></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>2
<FILENAME>ex99-1.htm
<TEXT>
<html><body>
<p>Example Corp Unveils Product X, an AI-Powered Analytics Assistant</p>
<p>SAN JOSE, Calif., Feb. 19, 2025 -- Example Corp (NASDAQ: EXMPL) today launched Product X, an
AI-powered assistant that helps finance teams automate forecasting and reporting.</p>
<p>Product X is generally available today to customers in North America, priced per user.</p>
<p>The company also reaffirmed its fiscal 2025 revenue guidance.</p>
</body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>GRAPHIC
<FILENAME>logo.jpg
<TEXT>begin 644 logo.jpg
M_]C_X``02D9)1@`!`0``
end
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>"""

NO_PRODUCT_SUBMISSION = SAMPLE_SUBMISSION.replace(
    "Unveils Product X, an AI-Powered Analytics Assistant", "Reports Fourth Quarter Results"
).replace("today launched Product X, an\nAI-powered assistant that helps finance teams automate forecasting and reporting.",
          "reported revenue of $1.2 billion.").replace(
    "Product X is generally available today to customers in North America, priced per user.",
    "Gross margin was 61 percent.")


@pytest.fixture(scope="module")
def nlp():
    return ex.load_nlp()


def test_split_submission_keeps_8k_and_ex99_only():
    docs = ex.split_submission(SAMPLE_SUBMISSION)
    assert [d["type"] for d in docs] == ["8-K", "EX-99.1"]
    assert "hidden xbrl" not in docs[0]["text"]          # iXBRL header removed
    assert "Product X" in docs[1]["text"]


def test_press_release_comes_first():
    text = ex.build_filing_text(ex.split_submission(SAMPLE_SUBMISSION))
    assert text.startswith("Example Corp Unveils Product X")


def test_launch_prefilter():
    assert ex.LAUNCH_RE.search(ex.build_filing_text(ex.split_submission(SAMPLE_SUBMISSION)))
    assert not ex.LAUNCH_RE.search(ex.build_filing_text(ex.split_submission(NO_PRODUCT_SUBMISSION)))


def test_rule_based_product_ner():
    assert "Product X" in ex.rule_based_products("Example Corp today launched Product X, an assistant.")
    assert "MacBook Pro" in ex.rule_based_products("Apple Inc. has announced the launch of its latest "
                                                   "MacBook Pro, featuring a new M3 processor.") or \
        "MacBook Pro" in ex.rule_based_products("Apple unveiled the new MacBook Pro today.")


def test_spacy_ner_assignment_example(nlp):
    passage, cands = ex.run_ner(nlp, "Apple Inc. has announced the launch of its latest MacBook Pro, "
                                     "featuring a new M3 processor.")
    assert "Apple Inc." in cands["ORG"]
    assert any("MacBook Pro" in p for p in cands["PRODUCT"])
    assert "MacBook Pro" in passage


def test_parse_llm_json_variants():
    good = '{"products": [{"product_name": "Product X", "description": "d", "evidence": "e"}]}'
    assert ex.parse_llm_json(good)[0]["product_name"] == "Product X"
    assert ex.parse_llm_json("<think>reasoning {not json}</think>\n```json\n" + good + "\n```")[0]
    assert ex.parse_llm_json("no json here") == []
    assert ex.parse_llm_json('{"products": []}') == []


def test_validation_grounding_and_length():
    text = "Example Corp today launched Product X, an AI-powered assistant."
    long_desc = "An AI-powered assistant " + "that automates forecasting and reporting " * 10
    valid, rejected = ex.validate(
        [{"product_name": "Product X", "description": long_desc, "evidence": ""},
         {"product_name": "Quantum Widget 9000", "description": "made up", "evidence": ""},
         {"product_name": "new product", "description": "x", "evidence": ""},
         {"product_name": "product x", "description": "duplicate", "evidence": ""}],
        text, "Example Corp")
    assert [v["new_product"] for v in valid] == ["Product X"]
    assert len(valid[0]["product_description"]) < 180
    assert any("not found" in r for r in rejected) and any("generic" in r for r in rejected)


def test_validation_drops_deals_and_other_companies_products():
    text = ("Example Corp will invest in Partner Inc and keep the Partner Hub platform open. "
            "Example Corp will host AI factories at River Campus. Example Corp launched Product X.")
    valid, rejected = ex.validate(
        [{"product_name": "Partner Hub platform", "announcement_type": "investment_or_acquisition",
          "offered_by_filer": False, "description": "d", "evidence": ""},
         {"product_name": "AI factories", "announcement_type": "facility_or_infrastructure",
          "offered_by_filer": True, "description": "d", "evidence": ""},
         {"product_name": "Partner Hub", "announcement_type": "new_product",
          "offered_by_filer": "false", "description": "d", "evidence": ""},
         {"product_name": "Product X", "announcement_type": "new_product",
          "offered_by_filer": True, "description": "A new assistant.", "evidence": ""}],
        text, "Example Corp")
    assert [v["new_product"] for v in valid] == ["Product X"]
    assert any("investment_or_acquisition" in r for r in rejected)
    assert any("facility_or_infrastructure" in r for r in rejected)
    assert any("not offered" in r for r in rejected)


def test_validation_drops_rows_the_model_itself_calls_an_acquisition():
    text = "Example Corp acquires Partner Inc and will keep the Partner Hub platform open. Probe Corp launched DAQ-9 data acquisition system."
    valid, rejected = ex.validate(
        [{"product_name": "Partner Hub platform", "announcement_type": "new_product", "offered_by_filer": True,
          "description": "Example Corp acquires Partner Inc, commits to keeping platform open.", "evidence": ""}],
        text, "Example Corp")
    assert valid == [] and "acquisition" in rejected[0]
    valid, _ = ex.validate(
        [{"product_name": "DAQ-9", "announcement_type": "new_product", "offered_by_filer": True,
          "description": "A new data acquisition system for labs.", "evidence": ""}], text, "Probe Corp")
    assert [v["new_product"] for v in valid] == ["DAQ-9"]


def test_trademark_symbols_are_grounded():
    assert ex.is_grounded("Copilot™ Studio", "Microsoft launched Copilot Studio today")


class FakeSec:
    def __init__(self, raw):
        self.raw = raw

    def get(self, url, cache_name=None):
        return self.raw


def test_end_to_end_with_mock_llm(nlp, tmp_path):
    cfg = ex.Config(user_agent="Test test@example.com", out_dir=tmp_path)
    prompts = []

    def fake_llm(prompt, schema=None):
        prompts.append(prompt)
        if schema is ex.VERIFY_SCHEMA:
            return json.dumps({"reason": "The text says Example Corp launched it.", "answer": "yes"})
        return json.dumps({"products": [{
            "product_name": "Product X",
            "announcement_type": "new_product",
            "offered_by_filer": True,
            "description": "An AI-powered assistant that helps finance teams automate forecasting and reporting.",
            "evidence": "today launched Product X"}]})

    filing = {"accession": "0000000001-25-000001", "filing_date": "2025-02-19",
              "acceptance_time": "2025-02-19T16:05:12.000Z", "items": "7.01,9.01"}
    rows, audit = ex.process_filing(filing, 1, "EXMPL", "Example Corp", FakeSec(SAMPLE_SUBMISSION),
                                    nlp, fake_llm, cfg)
    assert audit["status"] == "llm_products"
    assert "Product X" in prompts[0] and "EXMPL" in prompts[0]
    assert len(prompts) == 2 and 'Candidate product: "Product X"' in prompts[1]   # verify pass ran
    assert "Product X: yes" in audit["verify_notes"] and audit["llm_raw"].startswith("{")
    ex.append_rows(cfg.out_csv, ex.OUTPUT_COLUMNS, rows)
    df = pd.read_csv(cfg.out_csv)
    assert list(df.columns) == ex.OUTPUT_COLUMNS
    assert df.iloc[0].tolist()[:4] == ["Example Corp", "EXMPL", "2025-02-19", "Product X"]
    assert cfg.out_csv.read_text().splitlines()[1].startswith('"Example Corp","EXMPL"')  # quoted CSV
    table = ex.write_pipe_table(cfg.out_csv, cfg.table_txt)
    assert table.splitlines()[0].startswith("Company Name | Stock Name")

    # no-product filing never reaches the LLM
    rows2, audit2 = ex.process_filing({**filing, "accession": "x"}, 1, "EXMPL", "Example Corp",
                                      FakeSec(NO_PRODUCT_SUBMISSION), nlp, fake_llm, cfg)
    assert rows2 == [] and audit2["status"] == "skipped_no_launch_language" and len(prompts) == 2


def test_verifier_rejects_acquisition_candidate():
    text = ("Example Corp to Acquire Partner Inc. Example Corp today announced an agreement to acquire "
            "Partner Inc. Example Corp will keep the Partner Hub platform open to all developers. "
            "Separately, Example Corp launched Product X, an AI assistant for finance teams.")
    seen = {}

    def fake_llm(prompt, schema=None):
        assert schema is ex.VERIFY_SCHEMA
        name = prompt.split('Candidate product: "')[1].split('"')[0]
        seen[name] = prompt
        if name == "Partner Hub platform":
            return '{"reason": "It is the acquired company\'s platform, not a launch.", "answer": "no"}'
        return '{"reason": "Example Corp launched it.", "answer": "yes"}'

    cands = [{"new_product": "Partner Hub platform", "product_description": "d", "evidence": ""},
             {"new_product": "Product X", "product_description": "d", "evidence": ""}]
    kept, rejected, notes = ex.verify_products(cands, text, "Example Corp", fake_llm)
    assert [k["new_product"] for k in kept] == ["Product X"]
    assert "verifier said no" in rejected[0]
    assert "Example Corp to Acquire Partner Inc" in seen["Partner Hub platform"]   # headline included
    assert "keep the Partner Hub platform open" in seen["Partner Hub platform"]    # mention included


def test_parse_verify():
    assert ex.parse_verify('{"reason": "r", "answer": "yes"}') == (True, "r")
    assert ex.parse_verify('<think>hmm</think>{"reason": "deal", "answer": "no"}') == (False, "deal")
    assert ex.parse_verify("garbage")[0] is None


def test_timestamp_format():
    f = {"filing_date": "2025-02-19", "acceptance_time": "2025-02-19T16:05:12.000Z"}
    assert ex.format_filing_time(f, "date") == "2025-02-19"
    assert ex.format_filing_time(f, "timestamp") == "2025-02-19 16:05:12"


def test_submissions_json_filtering(tmp_path):
    cfg = ex.Config(user_agent="Test test@example.com", start_date="2025-01-01", end_date="2025-12-31",
                    cache_dir=tmp_path)
    sub = {"name": "EXAMPLE CORP", "filings": {"recent": {
        "form": ["8-K", "10-Q", "8-K", "8-K", "8-K/A"],
        "filingDate": ["2025-03-01", "2025-02-01", "2025-02-19", "2024-12-01", "2025-04-01"],
        "accessionNumber": ["a1", "a2", "a3", "a4", "a5"],
        "acceptanceDateTime": [""] * 5,
        "items": ["2.02,9.01", "", "7.01,9.01", "8.01", "7.01"],
    }}}
    out = ex.list_8k_filings(FakeSec(json.dumps(sub)), 1, cfg)
    assert [f["accession"] for f in out] == ["a3"]   # 8-K, in window, Item 7.01
