"""Unit tests for the rules engine and AI-output validation (no network)."""
import json
from pathlib import Path

import ai
import rules as R

S = Path(__file__).parent.parent / "data" / "samples"
A, B, C = (R.analyse((S / f).read_text()) for f in sorted(p.name for p in S.glob("*.txt")))


def sev(res, title_part):
    return next(f["severity"] for f in res["flags"] if title_part.lower() in f["title"].lower())


def test_grades():
    assert A["summary"]["grade"] == "Green" and B["summary"]["grade"] == "Red" and C["summary"]["grade"] == "Amber"


def test_extraction():
    t = B["terms"]
    assert t["rent"] == 45000 and t["deposit"] == 450000 and t["deposit_months"] == 10
    assert t["lockin_months"] == 11 and t["notice_tenant_months"] == 3 and t["escalation_pct"] == 15
    assert A["terms"]["refund_days"] == 15 and A["terms"]["entry_notice_hours"] == 24
    assert C["terms"]["late_fee"]["annual_pct"] == 36


def test_key_flags():
    assert sev(B, "deposit of 10") == "high" and sev(B, "forfeited") == "high"
    assert sev(B, "without notice") == "high" and sev(B, "late fee") == "high"
    assert sev(C, "hidden instruction") == "high" and sev(C, "12 hours") == "medium"
    assert sev(A, "deposit of 6") == "medium"


def test_threshold_is_flagged_as_borderline():
    """Q7: 6 vs 7 months' deposit flips medium -> high; the app marks it as near a threshold."""
    base = (S / "A_bengaluru_balanced.txt").read_text()
    seven = R.analyse(base.replace("Rs. 1,68,000", "Rs. 1,96,000"))
    assert sev(seven, "deposit of 7") == "high"
    assert next(f for f in A["flags"] if "deposit" in f["title"].lower())["near_threshold"]


def test_pii_masked_and_injection_removed():
    p = json.dumps(R.build_payload(C))
    assert "98220" not in p and "rohan.j@" not in p and "Sunita" not in p
    assert "Ignore all previous" not in p and "Removed by ClauseClear" in p
    pb = json.dumps(R.build_payload(B))
    assert "4521 7788 9012" not in pb and "[AADHAAR]" in pb


def test_input_validation():
    assert R.validate_text("too short")[1]
    assert R.validate_text("A recipe for biryani. " * 40)[1]  # long but not a rental agreement
    assert not R.validate_text((S / "A_bengaluru_balanced.txt").read_text())[1]


def test_paragraph_fallback_without_numbers():
    text = "\n\n".join(c["text"] for c in A["clauses"])
    res = R.analyse(text)
    assert len(res["clauses"]) >= 10 and res["terms"]["rent"] == 28000


def test_ai_cannot_soften_rule_flags():
    p = R.build_payload(B)
    fake = {"overview": "Fine.", "clauses": [{"id": "4", "plain": "Deposit of ₹4,50,000.", "for_you": "ok", "severity": "ok"},
                                             {"id": "Clause 10", "plain": "Visits.", "for_you": "", "severity": "high"},
                                             {"id": "99", "plain": "made up", "for_you": "", "severity": "ok"}],
            "top_concerns": ["Pay ₹12,345 extra (Clause 21)"], "questions_for_landlord": [], "confidence": "high"}
    e = ai.validate_explanation(fake, p)
    c4 = next(c for c in e["clauses"] if c["id"] == "4")
    assert c4["severity"] == "high" and c4["corrected"] and "4" in e["raised_by_rules"]
    assert [c["id"] for c in e["clauses"]] == ["4", "10"]          # unknown clause 99 dropped
    assert e["unverified"] == ["12,345"] and e["bad_citations"] == ["21"]


def test_figure_check_allows_lakh_and_rent_multiples():
    p = R.build_payload(B)
    assert ai.unverified_figures("Deposit 4.5 lakh, two months = ₹90,000, ₹1,000 a day", p) == []


def test_fallback_explanation():
    e = ai.fallback_explanation(R.build_payload(B), B)
    assert e["basic_mode"] and e["top_concerns"] and len(e["clauses"]) >= 14
