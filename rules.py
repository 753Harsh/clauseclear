"""ClauseClear rules engine: split a rental agreement into clauses, extract key terms, and flag risks for the tenant.

Everything here is deterministic Python. The AI only explains what this module finds; it cannot lower a severity.
"""
from __future__ import annotations

import hashlib
import math
import io
import re

SEV_ORDER = {"ok": 0, "low": 1, "medium": 2, "high": 3}
SEV_PENALTY = {"high": 15, "medium": 7, "low": 2, "ok": 0}
MAX_CHARS = 60_000
MIN_CHARS = 300

# ---------------------------------------------------------------- PII masking
MASK_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[AADHAAR]"),
    (re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b"), "[PHONE]"),
    (re.compile(r"\b\d{9,18}\b"), "[ACCOUNT_NO]"),
    (re.compile(r"\b(Mr|Mrs|Ms|Miss|Dr|Shri|Smt|Sri|Kum)\.?\s+([A-Z][a-z]+)(?:\s+[A-Z][a-z]+){0,2}"), r"\1. [NAME]"),
]


def mask_pii(text: str) -> str:
    """Hide names (after an honorific), e-mails, phones, PAN, Aadhaar and account numbers."""
    if not isinstance(text, str):
        return text
    for pattern, label in MASK_PATTERNS:
        text = pattern.sub(label, text)
    return text


INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\b.{0,40}\b(instruction|rule|prompt|above|previous|system)"
    r"|system prompt|developer mode|jailbreak|you are now|pretend to be|reveal your"
    r"|\b(AI|LLM|chatbot|GPT|Gemini|language model)\b.{0,40}\b(review|tool|assistant|reader)s?\b.{0,60}\b(report|say|state|rate|mark|do not mention)",
    re.IGNORECASE,
)


def looks_like_injection(text: str) -> bool:
    return bool(INJECTION.search(text or ""))


# ---------------------------------------------------------------- reading files
def read_upload(name: str, data: bytes) -> tuple[str, list[str]]:
    """Return (text, errors) for a .txt, .pdf or .docx upload."""
    name = (name or "").lower()
    try:
        if name.endswith(".pdf"):
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            if len(reader.pages) > 30:
                return "", ["The PDF has more than 30 pages. Upload only the agreement."]
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
            if len(text.strip()) < 50:
                return "", ["No text found in this PDF. It may be a scanned image; paste the text instead."]
        elif name.endswith(".docx"):
            import docx

            text = "\n\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
        elif name.endswith(".txt"):
            text = data.decode("utf-8", errors="replace")
        else:
            return "", ["Unsupported file type. Use .txt, .pdf or .docx."]
    except Exception as exc:  # noqa: BLE001
        return "", [f"Could not read the file ({exc.__class__.__name__})."]
    return text, []


RENTAL_WORDS = ["licensor", "licensee", "landlord", "tenant", "rent", "licence fee", "license fee",
                "security deposit", "premises", "lock-in", "lessor", "lessee", "notice"]


def validate_text(text: str) -> tuple[str, list[str], list[str]]:
    """Clean the agreement text. Returns (text, errors, warnings)."""
    errors, warnings = [], []
    text = (text or "").replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r"[  ]+", " ", text).strip()
    if len(text) < MIN_CHARS:
        errors.append(f"The text is too short ({len(text)} characters). Paste the full agreement (at least {MIN_CHARS}).")
        return text, errors, warnings
    if len(text) > MAX_CHARS:
        warnings.append(f"The agreement is long; only the first {MAX_CHARS:,} characters are analysed.")
        text = text[:MAX_CHARS]
    hits = sum(1 for w in RENTAL_WORDS if w in text.lower())
    if hits < 3:
        errors.append("This doesn't look like a rental / leave-and-license agreement (key terms such as rent, "
                      "deposit, licensor or tenant are missing). ClauseClear only reviews residential rental agreements.")
    return text, errors, warnings


# ---------------------------------------------------------------- clause splitting
HEAD = re.compile(r"^\s*(?:clause\s+)?(\d{1,2})\s*[.):-]\s+", re.IGNORECASE | re.MULTILINE)


def split_clauses(text: str) -> list[dict]:
    """Split on numbered headings ("1.", "2)", "Clause 3 -"). Falls back to paragraphs."""
    marks = list(HEAD.finditer(text))
    clauses = []
    if len(marks) >= 3:
        pre = text[: marks[0].start()].strip()
        if pre:
            clauses.append({"id": "P", "text": pre})
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            clauses.append({"id": m.group(1), "text": text[m.end():end].strip()})
    else:
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        clauses = [{"id": str(i + 1), "text": p} for i, p in enumerate(paras)]
    seen = {}
    for c in clauses:  # make ids unique if numbering repeats
        if c["id"] in seen:
            seen[c["id"]] += 1
            c["id"] = f"{c['id']}.{seen[c['id']]}"
        else:
            seen[c["id"]] = 1
        m = re.match(r"([A-Z][A-Za-z &/-]{2,40}?)\s*[:.-]\s", c["text"])
        c["title"] = m.group(1).strip() if m else (c["text"][:40].rsplit(" ", 1)[0] + "…")
        c["type"] = classify(c)
    if clauses and clauses[0]["id"] == "P":
        clauses[0]["title"], clauses[0]["type"] = "Parties and preamble", "parties"
    return clauses


TYPES = [  # (type, keywords in title, keywords in body)
    ("injection", [], []),
    ("deposit", ["deposit"], ["security deposit", "refundable deposit"]),
    ("lockin", ["lock"], ["lock-in", "lock in"]),
    ("termination", ["terminat", "notice", "vacat"], ["terminate", "notice to vacate"]),
    ("escalation", ["escalat", "increase", "revision", "enhance"], ["increased by", "escalation"]),
    ("late_fee", ["late", "delay", "interest"], ["late fee", "delayed payment", "per day of delay"]),
    ("entry", ["access", "entry", "inspect", "visit"], ["enter the premises", "inspect the premises"]),
    ("painting", ["paint", "clean"], ["painting"]),
    ("repairs", ["repair", "maintenance and repair"], ["structural repair", "repairs"]),
    ("maintenance", ["maintenance charge", "society"], ["maintenance charges"]),
    ("rent", ["rent", "licence fee", "license fee", "compensation"], ["per month", "monthly"]),
    ("fees", ["brokerage", "fee", "charges"], ["non-refundable"]),
    ("utilities", ["utilit", "electricity"], ["electricity"]),
    ("use", ["use", "guest", "sublet", "pets"], ["sublet", "guests"]),
    ("registration", ["stamp", "registration"], ["stamp duty"]),
    ("dispute", ["dispute", "jurisdiction", "arbitrat", "law"], ["arbitrat", "jurisdiction"]),
    ("term", ["period", "term", "duration"], ["months commencing", "period of"]),
    ("premises", ["premises", "property"], ["flat no"]),
]


def classify(c: dict) -> str:
    if looks_like_injection(c["text"]):
        return "injection"
    title, body = c.get("title", "").lower(), c["text"].lower()
    for t, tk, bk in TYPES[1:]:
        if any(k in title for k in tk):
            return t
    for t, tk, bk in TYPES[1:]:
        if any(k in body for k in bk):
            return t
    return "other"


# ---------------------------------------------------------------- number helpers
WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "thirty": 30, "a": 1}
AMOUNT = re.compile(r"(?:₹|Rs\.?|INR|Rupees)\s*([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?)?", re.IGNORECASE)
DURATION = re.compile(r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|thirty|a)\b"
                      r"(?:\s*\(\w+\))?\s*(?:\(\d+\)\s*)?(month|day|hour|week|year)s?(?:'s|’s|')?", re.IGNORECASE)


def amounts(text: str) -> list[float]:
    out = []
    for num, lakh in AMOUNT.findall(text):
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        out.append(v * 100_000 if lakh else v)
    return out


def durations(text: str) -> list[tuple[float, str]]:
    """[(value, unit)] e.g. 'three (3) months' -> (3, 'month')."""
    res = []
    for n, unit in DURATION.findall(text):
        v = WORDNUM.get(n.lower()) if not n.isdigit() else int(n)
        if v is not None:
            res.append((float(v), unit.lower()))
    return res


def to_months(v: float, unit: str) -> float:
    return {"month": v, "day": v / 30, "week": v / 4.3, "year": v * 12, "hour": v / 720}[unit]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z])", text) if s.strip()]


def of_type(clauses, *types):
    return [c for c in clauses if c["type"] in types]


# ---------------------------------------------------------------- extraction
def extract_terms(clauses: list[dict]) -> dict:
    t: dict = {"rent": None, "deposit": None, "deposit_months": None, "refund_days": None, "lockin_months": None,
               "notice_tenant_months": None, "notice_landlord_months": None, "escalation_pct": None,
               "entry_notice_hours": None, "term_months": None, "late_fee": None}
    src: dict = {}
    for c in of_type(clauses, "rent"):
        a = [x for x in amounts(c["text"]) if x >= 1000]
        if a and t["rent"] is None:
            t["rent"], src["rent"] = a[0], c["id"]
    for c in of_type(clauses, "deposit"):
        a = [x for x in amounts(c["text"]) if x >= 1000]
        if a:
            t["deposit"], src["deposit"] = max(a), c["id"]
        elif t["rent"]:
            m = re.search(DURATION.pattern + r"\s*(?:rent|licence fee|license fee)", c["text"], re.I)
            if m:
                d = durations(m.group(0))
                if d:
                    t["deposit"], src["deposit"] = d[0][0] * t["rent"], c["id"]
        m = re.search(r"within\s+(\d{1,3}|\w+)\s*(?:\(\d+\)\s*)?days", c["text"], re.I)
        if m:
            n = m.group(1)
            t["refund_days"] = int(n) if n.isdigit() else WORDNUM.get(n.lower())
    if t["rent"] and t["deposit"]:
        t["deposit_months"] = round(t["deposit"] / t["rent"], 1)
    for c in of_type(clauses, "lockin"):
        d = [to_months(v, u) for v, u in durations(c["text"]) if u in ("month", "year")]
        if d:
            t["lockin_months"], src["lockin"] = d[0], c["id"]
    for c in of_type(clauses, "termination", "lockin"):
        for s in sentences(c["text"]):
            low = s.lower()
            if "notice" not in low:
                continue
            d = [to_months(v, u) for v, u in durations(s)]
            if not d:
                continue
            if "either party" in low or "both parties" in low:
                t["notice_tenant_months"] = t["notice_tenant_months"] or d[0]
                t["notice_landlord_months"] = t["notice_landlord_months"] or d[0]
            elif re.search(r"licensee|tenant|lessee", low) and not re.search(r"licensor (may|can|shall) terminat|landlord (may|can)", low):
                t["notice_tenant_months"] = t["notice_tenant_months"] or d[0]
            elif re.search(r"licensor|landlord|lessor|owner", low):
                t["notice_landlord_months"] = t["notice_landlord_months"] or round(d[0], 2)
    for c in of_type(clauses, "escalation"):
        m = re.search(r"(\d{1,2}(?:\.\d)?)\s*%", c["text"])
        if m:
            t["escalation_pct"] = float(m.group(1))
    for c in of_type(clauses, "entry"):
        d = [v * (24 if u == "day" else 1) for v, u in durations(c["text"]) if u in ("hour", "day")]
        if d:
            t["entry_notice_hours"] = d[0]
    for c in of_type(clauses, "term"):
        d = [to_months(v, u) for v, u in durations(c["text"]) if u in ("month", "year")]
        if d:
            t["term_months"] = d[0]
    for c in of_type(clauses, "late_fee"):
        a = amounts(c["text"])
        pct = re.search(r"(\d{1,2}(?:\.\d+)?)\s*%\s*(?:per|a|p\.?)\s*(month|annum|year|day|m\b)", c["text"], re.I)
        if a and re.search(r"per day|each day|daily", c["text"], re.I):
            t["late_fee"] = {"kind": "per_day", "amount": a[0]}
        elif pct:
            p, per = float(pct.group(1)), pct.group(2).lower()
            annual = p * 12 if per.startswith("m") else p * 365 if per == "day" else p
            t["late_fee"] = {"kind": "interest", "pct": p, "per": per, "annual_pct": annual}
    t["sources"] = src
    return t


# ---------------------------------------------------------------- risk rules
BENCHMARKS = [
    ("Security deposit", "Up to 2 months' rent is low risk (Model Tenancy Act, 2021 cap for residential). 2–6 months is common in some cities (medium). Above 6 months is high."),
    ("Deposit refund", "Refund within 30 days of handing back the keys, deductions only for actual dues/damage with bills. Refund tied to finding a new tenant, or forfeiture, is high risk."),
    ("Lock-in", "Up to 6 months and binding on both sides is normal. Longer, or binding only the tenant, is medium; full-term lock-in is high."),
    ("Notice period", "One month each way is common. Tenant notice above 2 months, or a shorter notice for the landlord than for the tenant, is medium; landlord ending it 'at any time' or in under 15 days is high."),
    ("Rent escalation", "5–10% on renewal is common. Above 10% is medium, 15% or 'as decided by the landlord' is high."),
    ("Late fee", "Interest up to 24% a year (2% a month) is low risk. More is medium. A flat per-day fee above 1% of monthly rent is high."),
    ("Landlord entry", "At least 24 hours' notice (Model Tenancy Act, 2021). Less is medium; 'any time without notice' is high."),
    ("Repairs", "Tenant pays small day-to-day repairs; structural repairs and seepage are the owner's. Tenant paying for all or structural repairs is high."),
    ("Painting / cleaning", "Actual cost, only for damage beyond normal wear and tear, with bills. A fixed deduction regardless of condition is medium."),
    ("Extra fees", "Non-refundable 'processing' or 'maintenance' fees paid to the landlord's agent are medium; check what you get for them."),
    ("Disputes", "Courts or a neutral arbitrator in the city of the property. An arbitrator chosen only by the landlord is medium."),
    ("Stamp duty", "Usually shared or as agreed. Tenant paying all of it is low risk but negotiable."),
    ("Text aimed at AI", "Agreements should not contain instructions to AI tools. Any such text is high risk and is removed before the AI sees it."),
]


def _flag(flags, clause, sev, title, finding, ask, check, near=False):
    flags.append({"clause_id": clause["id"] if clause else None, "clause_title": clause["title"] if clause else "Missing",
                  "severity": sev, "title": title, "finding": finding, "ask": ask, "check": check, "near_threshold": near})


def inr(x: float) -> str:
    x = round(float(x))
    s = str(abs(x))
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return "₹" + (",".join(groups + [tail]) if groups else tail)


def check_risks(clauses: list[dict], terms: dict) -> list[dict]:
    f: list[dict] = []
    by = {c["id"]: c for c in clauses}
    first = lambda *ts: next(iter(of_type(clauses, *ts)), None)  # noqa: E731
    dep = by.get(terms["sources"].get("deposit")) or first("deposit")

    # injection
    for c in of_type(clauses, "injection"):
        _flag(f, c, "high", "Hidden instruction aimed at AI tools",
              "This clause tries to tell an AI reviewer what to say. It has no place in a rental agreement and is "
              "removed before the text is sent to Gemini.",
              "Ask why this text is in the agreement and have it struck out before signing.", "Text aimed at AI")

    # deposit
    m = terms["deposit_months"]
    if m is not None:
        sev = "ok" if m <= 2 else "medium" if m <= 6 else "high"
        near = any(abs(m - th) <= 0.5 for th in (2, 6))
        _flag(f, dep, sev, f"Security deposit of {m:g} months' rent",
              f"Deposit {inr(terms['deposit'])} against rent {inr(terms['rent'])} = {m:g} months.",
              "Ask to reduce the deposit to 2–3 months, or pay part of it as post-dated cheques." if sev != "ok" else
              "Fine as written.", "Security deposit", near)
    elif dep:
        _flag(f, dep, "low", "Deposit amount not detected", "ClauseClear could not read the deposit or rent amount.",
              "Check the deposit amount manually.", "Security deposit")
    if dep:
        low = dep["text"].lower()
        if re.search(r"forfeit|non[- ]refundable", low):
            _flag(f, dep, "high", "Deposit can be forfeited",
                  "The clause lets the landlord keep the deposit (forfeiture), not just deduct actual dues.",
                  "Replace forfeiture with deduction of actual unpaid dues and damage, with bills.", "Deposit refund")
        if re.search(r"(new|another|next) (tenant|licensee)|re-?let", low):
            _flag(f, dep, "high", "Refund depends on the landlord finding a new tenant",
                  "Your money could be held indefinitely; the refund date is outside your control.",
                  "Ask for a fixed refund date, e.g. within 15–30 days of handing over the keys.", "Deposit refund")
        elif terms["refund_days"] is None:
            _flag(f, dep, "medium", "No refund timeline for the deposit",
                  "The agreement does not say when the deposit will be returned.",
                  "Add: 'refunded within 30 days of vacating'.", "Deposit refund")
        elif terms["refund_days"] > 30:
            _flag(f, dep, "medium", f"Deposit refund takes {terms['refund_days']} days",
                  f"Refund within {terms['refund_days']} days is slower than the usual 30.", "Ask for 30 days or less.",
                  "Deposit refund")
    elif terms["rent"]:
        _flag(f, None, "medium", "No security deposit clause found",
              "Deposit terms (amount, refund date, deductions) should be in writing.",
              "Ask for a written deposit clause with a refund date.", "Deposit refund")

    # lock-in
    lk = first("lockin")
    if lk and terms["lockin_months"] is not None:
        L = terms["lockin_months"]
        term = terms["term_months"] or 11
        one_sided = not re.search(r"both parties|either party|mutual", lk["text"], re.I)
        sev = "high" if L >= term else "medium" if (L > 6 or one_sided) else "ok"
        _flag(f, lk, sev, f"Lock-in of {L:g} months" + (" (binds only the tenant)" if one_sided else " (both sides)"),
              f"You cannot leave for {L:g} of the {term:g} months" + (" while the landlord is not bound the same way." if one_sided else "."),
              "Ask for a lock-in of 6 months or less that binds both parties." if sev != "ok" else "Fine as written.",
              "Lock-in", abs(L - 6) <= 0.5 and sev != "high")
        if re.search(r"balance|remaining|entire|unexpired", lk["text"], re.I):
            _flag(f, lk, "medium", "Penalty = rent for the rest of the lock-in",
                  "Leaving early means paying rent for all remaining lock-in months.",
                  "Cap the penalty at one month's rent.", "Lock-in")

    # notice
    term_c = first("termination") or lk
    nt, nl = terms["notice_tenant_months"], terms["notice_landlord_months"]
    land_any = term_c and re.search(r"(licensor|landlord|owner)[^.]{0,60}(at any time|forthwith|without (any )?(prior )?notice|immediately)", term_c["text"], re.I)
    if land_any or (nl is not None and nl < 0.5):
        _flag(f, term_c, "high", "Landlord can end the agreement almost immediately",
              "The landlord can terminate at any time" + (f" with only {round(nl * 30)} days' notice" if nl else "") + ".",
              "Ask for at least one month's notice from the landlord, and none during the lock-in.", "Notice period")
    if nt is not None and nt > 2:
        _flag(f, term_c, "medium", f"Tenant notice of {nt:g} months",
              f"You must give {nt:g} months' notice, longer than the usual one month.", "Ask for one month's notice.",
              "Notice period")
    if nt is not None and nl is not None and nl < nt and not (nl < 0.5):
        _flag(f, term_c, "medium", "Notice periods are unequal",
              f"You give {nt:g} month(s); the landlord gives {nl:g}.", "Ask for the same notice period for both sides.",
              "Notice period")
    if term_c and nl is None and not land_any:
        _flag(f, term_c, "low", "Landlord's notice period not stated",
              "The agreement does not say how much notice the landlord must give.",
              "Add a notice period for the landlord (at least one month).", "Notice period")

    # escalation
    esc = first("escalation")
    if esc:
        p = terms["escalation_pct"]
        disc = re.search(r"as decided by|sole discretion|at the discretion", esc["text"], re.I)
        if disc or (p is not None and p >= 15):
            sev = "high"
        elif p is not None and p > 10:
            sev = "medium"
        else:
            sev = "ok"
        _flag(f, esc, sev, f"Rent increase of {p:g}%" + (" or at the landlord's discretion" if disc else "") if p is not None
              else "Rent increase at the landlord's discretion",
              "On renewal the rent can go up " + (f"by {p:g}%" if p is not None else "") + (" or by any amount the landlord decides." if disc else "."),
              "Ask to fix the increase at 5–10% and remove 'as decided by the landlord'." if sev != "ok" else "Fine as written.",
              "Rent escalation", p is not None and abs(p - 10) <= 1)

    # late fee
    lf, lc = terms["late_fee"], first("late_fee")
    if lf and lc:
        if lf["kind"] == "per_day":
            share = lf["amount"] / terms["rent"] * 100 if terms["rent"] else None
            sev = "high" if share is None or share > 1 else "medium"
            _flag(f, lc, sev, f"Late fee of {inr(lf['amount'])} per day",
                  f"{inr(lf['amount'])} a day" + (f" is {share:.1f}% of the monthly rent per day; a week's delay costs {inr(lf['amount'] * 7)}." if share else "."),
                  "Ask for simple interest of up to 18–24% a year instead of a flat daily fee.", "Late fee")
        else:
            a = lf["annual_pct"]
            sev = "ok" if a <= 24 else "medium"
            _flag(f, lc, sev, f"Late-payment interest of {lf['pct']:g}% per {lf['per']}",
                  f"That is about {a:g}% a year.", "Ask to cap it at 18–24% a year." if sev != "ok" else "Fine as written.",
                  "Late fee")

    # entry
    en = first("entry")
    if en:
        h = terms["entry_notice_hours"]
        if re.search(r"any ?time|without (any )?(prior )?notice", en["text"], re.I):
            _flag(f, en, "high", "Landlord can enter without notice",
                  "The landlord or agents may enter at any time without telling you.",
                  "Ask for at least 24 hours' written notice and visits at reasonable hours.", "Landlord entry")
        elif h is not None and h < 24:
            _flag(f, en, "medium", f"Only {h:g} hours' notice before landlord visits",
                  f"{h:g} hours is less than the 24 hours suggested by the Model Tenancy Act, 2021.",
                  "Ask for 24 hours' notice.", "Landlord entry")
        else:
            _flag(f, en, "ok", "Landlord visits need prior notice", "Reasonable notice is required before visits.",
                  "Fine as written.", "Landlord entry")
    else:
        _flag(f, None, "low", "No clause on landlord visits",
              "The agreement does not say when the landlord may enter.", "Add: visits only with 24 hours' notice.",
              "Landlord entry")

    # repairs
    rp = first("repairs")
    if rp:
        low = rp["text"].lower()
        if re.search(r"all repairs|structural[^.]{0,80}(licensee|tenant)|(licensee|tenant)[^.]{0,120}structural", low) and not re.search(r"structural[^.]{0,60}(borne by|done by|by) the (licensor|landlord)", low):
            _flag(f, rp, "high", "Tenant pays for all repairs, including structural",
                  "Structural repairs, seepage and appliance replacement are normally the owner's cost.",
                  "Limit your share to minor repairs (e.g. up to ₹1,000 each).", "Repairs")
        else:
            _flag(f, rp, "ok", "Repairs split fairly", "Minor repairs are yours; major/structural are the owner's.",
                  "Fine as written.", "Repairs")
    else:
        _flag(f, None, "medium", "No repairs clause", "Who pays for repairs is not stated.",
              "Add who pays for minor and structural repairs.", "Repairs")

    # painting
    pt = first("painting")
    if pt and re.search(r"irrespective|in any case|regardless|one month'?s? rent|fixed", pt["text"], re.I):
        _flag(f, pt, "medium", "Fixed painting deduction regardless of condition",
              "A fixed amount is cut from your deposit even if the flat is in good condition.",
              "Ask for actual cost, only for damage beyond normal wear and tear, with bills.", "Painting / cleaning")

    # fees
    for c in of_type(clauses, "fees"):
        if re.search(r"non[- ]refundable", c["text"], re.I):
            a = amounts(c["text"])
            _flag(f, c, "medium", "Non-refundable fee" + (f" of {inr(a[0])}" if a else ""),
                  "You pay a fee you will not get back" + (f" ({inr(a[0])})" if a else "") + ", in addition to rent and deposit.",
                  "Ask what the fee covers and whether it can be waived or adjusted against rent.", "Extra fees")

    # maintenance charges at discretion
    for c in of_type(clauses, "maintenance"):
        if re.search(r"as decided by|from time to time|sole discretion", c["text"], re.I):
            _flag(f, c, "medium", "Maintenance charges can change at the landlord's discretion",
                  "Society charges are open-ended and set by the landlord.",
                  "Fix the amount, or link it to the society's actual bill.", "Extra fees")

    # use restrictions
    for c in of_type(clauses, "use"):
        if re.search(r"no guests?|guests? shall not|without (the )?(prior )?(written )?permission", c["text"], re.I):
            _flag(f, c, "low", "Restrictions on guests", "Guests need the landlord's permission to stay overnight.",
                  "Ask to allow family and short guest stays.", "Extra fees")

    # registration
    for c in of_type(clauses, "registration"):
        if re.search(r"entirely|solely|wholly", c["text"], re.I):
            _flag(f, c, "low", "You pay all stamp duty and registration",
                  "These costs are often shared.", "Ask to split them 50:50.", "Stamp duty")

    # disputes
    for c in of_type(clauses, "dispute"):
        if re.search(r"(arbitrator|arbitrators)[^.]{0,40}appointed by the (licensor|landlord|owner)", c["text"], re.I):
            _flag(f, c, "medium", "Landlord alone chooses the arbitrator",
                  "A dispute would be decided by someone the landlord picks.",
                  "Ask for a mutually agreed arbitrator, or courts in the city of the property.", "Disputes")
    return f


def score(flags: list[dict]) -> dict:
    counts = {s: sum(1 for x in flags if x["severity"] == s) for s in ("high", "medium", "low", "ok")}
    penalty = sum(SEV_PENALTY[x["severity"]] for x in flags)
    val = round(100 * math.exp(-penalty / 70))  # smooth: each extra problem costs a little less than the last
    grade = "Green" if val >= 75 else "Amber" if val >= 40 else "Red"
    verdict = {"Green": "Broadly fair to the tenant", "Amber": "Negotiate before signing",
               "Red": "One-sided: do not sign as is"}[grade]
    return {"score": val, "grade": grade, "verdict": verdict, "counts": counts}


def analyse(text: str) -> dict:
    clauses = split_clauses(text)
    terms = extract_terms(clauses)
    flags = check_risks(clauses, terms)
    flags.sort(key=lambda x: (-SEV_ORDER[x["severity"]], x["clause_id"] or "zz"))
    return {"clauses": clauses, "terms": terms, "flags": flags, "summary": score(flags),
            "hash": hashlib.sha1(text.encode()).hexdigest()[:12]}


def clause_severity(flags: list[dict]) -> dict:
    out: dict = {}
    for x in flags:
        cid = x["clause_id"]
        if cid and SEV_ORDER[x["severity"]] > SEV_ORDER.get(out.get(cid, "ok"), 0):
            out[cid] = x["severity"]
        out.setdefault(cid, x["severity"])
    return out


def build_payload(result: dict) -> dict:
    """Fact sheet sent to Gemini: masked clause text, rule findings and extracted terms. Injection text is removed."""
    sev = clause_severity(result["flags"])
    clauses = []
    for c in result["clauses"]:
        if c["type"] == "injection":
            text = "[Removed by ClauseClear: this clause contained instructions addressed to AI tools.]"
        else:
            text = mask_pii(c["text"])[:900]
        clauses.append({"id": c["id"], "title": mask_pii(c["title"]), "type": c["type"], "text": text,
                        "rule_severity": sev.get(c["id"], "ok")})
    t = {k: v for k, v in result["terms"].items() if k != "sources"}
    return {
        "perspective": "tenant (licensee)",
        "clauses": clauses,
        "extracted_terms": t,
        "rule_flags": [{k: x[k] for k in ("clause_id", "severity", "title", "finding", "ask")} for x in result["flags"]
                       if x["severity"] != "ok"],
        "overall": result["summary"],
    }
