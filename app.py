"""ClauseClear: AI rental-agreement explainer for tenants (AI for Managers end-term project, Use Case #19)."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import ai
import rules as R

st.set_page_config(page_title="ClauseClear · Rental Agreement Explainer", page_icon="📜", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 3.2rem; max-width: 1250px;}
.cc-hero {background: linear-gradient(120deg,#1e1b4b 0%,#3730a3 60%,#4f46e5 100%); color:#fff; padding:18px 24px;
          border-radius:14px; margin-bottom:14px;}
.cc-hero h1 {color:#fff; font-size:1.75rem; margin:0 0 2px 0; padding:0;}
.cc-hero p {color:#e0e7ff; margin:0; font-size:0.95rem;}
.cc-kpi {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:12px 16px; height:100%;}
.cc-kpi .l {font-size:.8rem; opacity:.7;} .cc-kpi .v {font-size:1.35rem; font-weight:700; line-height:1.3;} .cc-kpi .s {font-size:.8rem; opacity:.8;}
.cc-pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; margin-right:6px;}
.cc-high {background:#fee2e2; color:#7f1d1d;} .cc-medium {background:#fef3c7; color:#78350f;}
.cc-low {background:#e0f2fe; color:#0c4a6e;} .cc-ok {background:#dcfce7; color:#14532d;}
.cc-flag {border:1px solid rgba(128,128,128,.25); border-left:6px solid #999; border-radius:10px; padding:10px 14px; margin-bottom:8px;}
.cc-flag.high {border-left-color:#dc2626;} .cc-flag.medium {border-left-color:#d97706;} .cc-flag.low {border-left-color:#0284c7;} .cc-flag.ok {border-left-color:#16a34a;}
.cc-flag .t {font-weight:650;} .cc-flag .f {font-size:.9rem; margin-top:2px;} .cc-flag .a {font-size:.86rem; opacity:.85; margin-top:3px;}
.cc-small {font-size:0.82rem; opacity:.75;}
</style>
""",
    unsafe_allow_html=True,
)
SEV_LABEL = {"high": "High risk", "medium": "Medium", "low": "Low", "ok": "OK"}
SAMPLES = {
    "A · Bengaluru, balanced": "A_bengaluru_balanced.txt",
    "B · Gurugram, landlord-heavy": "B_gurugram_landlord_heavy.txt",
    "C · Pune, with a hidden trick": "C_pune_hidden_instruction.txt",
}
HERE = Path(__file__).parent


def pill(sev: str, text: str | None = None) -> str:
    return f'<span class="cc-pill cc-{sev}">{text or SEV_LABEL[sev]}</span>'


def fmt_notice(m):
    if not m:
        return "?"
    return f"{round(m * 30)} days" if m < 1 else f"{m:g} mo"


def kpi(col, label, value, sub="", color="inherit"):
    col.markdown(f'<div class="cc-kpi"><div class="l">{label}</div><div class="v" style="color:{color}">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ state
ss = st.session_state
for k, v in {"text": "", "source": "", "result": None, "expl": None, "expl_key": None, "chat": [], "ai_calls": 0,
             "input_errors": [], "input_warnings": [], "user_key": "", "lang": "English", "ai_nonce": 0}.items():
    ss.setdefault(k, v)


# ------------------------------------------------------------------ AI setup
def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.environ.get(name, "")
    except Exception:  # noqa: BLE001
        return os.environ.get(name, "")


@st.cache_resource(show_spinner=False)
def get_client(key: str, models: tuple):
    return ai.GeminiClient(key, list(models))


@st.cache_data(ttl=600, show_spinner=False)
def check_ai(key: str, models: tuple, nonce: int):
    try:
        return {"ok": True, "model": get_client(key, models).ping(), "msg": ""}
    except ai.AIError as e:
        return {"ok": False, "model": None, "msg": e.message, "kind": e.kind}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "model": None, "msg": f"AI setup failed ({e.__class__.__name__}).", "kind": "network"}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_explain(key: str, models: tuple, payload_json: str, lang: str):
    """Same agreement + language → same explanation for an hour (saves free-tier quota, stable answers)."""
    return get_client(key, models).explain(json.loads(payload_json), lang)


API_KEY = ss.user_key.strip() or secret("GEMINI_API_KEY")
m = secret("GEMINI_MODEL")
MODELS = tuple(([m] if m else []) + [x for x in ai.DEFAULT_MODELS if x != m])
status = check_ai(API_KEY, MODELS, ss.ai_nonce) if API_KEY else {"ok": False, "msg": "No Gemini API key configured.", "kind": "key"}
if ss.ai_calls >= ai.MAX_CALLS_PER_SESSION:
    status = {"ok": False, "msg": f"Session limit of {ai.MAX_CALLS_PER_SESSION} AI calls reached (protects the free quota)."}
AI_ON = status["ok"]


# ------------------------------------------------------------------ analysis
def load_text(text: str, source: str):
    text, errs, warns = R.validate_text(text)
    ss.input_errors, ss.input_warnings = errs, warns
    if errs:
        return
    ss.text, ss.source = text, source
    ss.result = R.analyse(text)
    ss.expl, ss.expl_key, ss.chat = None, None, []


def use_sample():
    name = ss.sample_pick
    load_text((HERE / "data" / "samples" / SAMPLES[name]).read_text(encoding="utf-8"), f"Sample agreement {name} (fictional)")


def use_paste():
    load_text(ss.get("paste_box", ""), "Pasted text")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 📜 ClauseClear")
    if AI_ON:
        st.markdown(f'<span class="cc-pill cc-ok">● AI mode</span> Gemini `{status["model"]}`', unsafe_allow_html=True)
    else:
        st.markdown('<span class="cc-pill cc-medium">● Basic mode (no AI)</span>', unsafe_allow_html=True)
        st.caption(status["msg"] + " Risk checks still work; explanations use templates.")
        if st.button("Retry AI mode"):
            ss.ai_nonce += 1
            check_ai.clear()
            st.rerun()
    calls_slot = st.empty()
    st.divider()
    st.radio("Explain in", list(ai.LANGS), key="lang", horizontal=True)
    st.divider()
    with st.expander("🔒 Privacy & AI disclosure"):
        st.markdown(
            "- Explanations and chat use **Google Gemini**, a third-party AI. The agreement text is sent to Google's API.\n"
            "- Before sending, names after Mr/Ms/Smt etc., phone numbers, e-mails, PAN, Aadhaar and account numbers are "
            "**masked**, and any text addressed to AI tools is removed. See *What is sent to Gemini* in tab ①.\n"
            "- The risk flags and score are calculated by the app's rules, **not** by the AI.\n"
            "- Nothing is stored after you close the tab. Free-tier Gemini may use inputs to improve Google's products."
        )
    with st.expander("⚖️ Not legal advice"):
        st.markdown("ClauseClear explains common rental terms against simple market benchmarks. It is not a lawyer and "
                    "cannot tell you whether a clause is enforceable. For disputes or high-value agreements, consult a "
                    "lawyer or your state's Rent Authority.")
    with st.expander("Use your own Gemini key (optional)"):
        st.text_input("Gemini API key", type="password", key="user_key", help="Only used in this browser session.")
    if st.button("Reset session"):
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()

st.markdown(
    """<div class="cc-hero"><h1>ClauseClear: Rental Agreement Explainer</h1>
<p>Paste or upload an 11-month rental (leave-and-license) agreement. ClauseClear flags risky clauses for the tenant,
explains every clause in plain English or Hindi, and tells you what to ask the landlord before you sign.</p></div>""",
    unsafe_allow_html=True,
)

t1, t2, t3, t4, t5 = st.tabs(["① Agreement", "② Risk report", "③ Plain-language explainer", "④ Ask ClauseClear",
                              "⑤ Benchmarks & method"])
res = ss.result
payload = R.build_payload(res) if res else None

# ------------------------------------------------------------------ tab 1
with t1:
    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        st.markdown("**Try a sample agreement** (fictional)")
        st.radio("Sample", list(SAMPLES), key="sample_pick", label_visibility="collapsed")
        st.button("Load sample", type="primary", on_click=use_sample)
        st.markdown("**…or upload your own** (.txt, .pdf, .docx · max 5 MB)")
        up = st.file_uploader("Agreement file", type=["txt", "pdf", "docx"], label_visibility="collapsed")
        if up is not None and st.button("Analyse uploaded file"):
            text, errs = R.read_upload(up.name, up.getvalue())
            if errs:
                ss.input_errors = errs
            else:
                load_text(text, f"Uploaded: {up.name}")
            st.rerun()
    with c2:
        st.markdown("**…or paste the agreement text**")
        st.text_area("Agreement text", key="paste_box", height=200, label_visibility="collapsed",
                     placeholder="Paste the full agreement here, including the numbered clauses…")
        st.button("Analyse pasted text", on_click=use_paste)
    for e in ss.input_errors:
        st.error(e)
    for w in ss.input_warnings:
        st.warning(w)
    if res:
        st.success(f"**Loaded:** {ss.source} · {len(res['clauses'])} clauses found · "
                   f"{res['summary']['counts']['high']} high and {res['summary']['counts']['medium']} medium risks. "
                   "Open tab ② for the risk report.")
        with st.expander("Clauses as ClauseClear read them"):
            st.dataframe(pd.DataFrame([{"Clause": c["id"], "Title": c["title"], "Type": c["type"],
                                        "Text": c["text"][:160]} for c in res["clauses"]]),
                         hide_index=True, use_container_width=True)
        with st.expander("🔒 What is sent to Gemini (masked)"):
            st.caption("Personal details are masked and any text addressed to AI tools is removed before the API call.")
            st.code("\n\n".join(f"{c['id']}. {c['text']}" for c in payload["clauses"]), language=None)

# ------------------------------------------------------------------ tab 2
with t2:
    if not res:
        st.info("Load an agreement in tab ① first.")
    else:
        s, tm = res["summary"], res["terms"]
        color = {"Green": "#15803d", "Amber": "#b45309", "Red": "#b91c1c"}[s["grade"]]
        k = st.columns(5)
        kpi(k[0], "Tenant-friendliness score", f"{s['score']}/100", f"{s['grade']} · {s['verdict']}", color)
        kpi(k[1], "Risks found", f"{s['counts']['high']} high · {s['counts']['medium']} med", f"{s['counts']['low']} low")
        kpi(k[2], "Monthly rent", R.inr(tm["rent"]) if tm["rent"] else "not found", f"{tm['term_months'] or '?':g} months term" if tm["term_months"] else "")
        kpi(k[3], "Security deposit", R.inr(tm["deposit"]) if tm["deposit"] else "not found",
            f"= {tm['deposit_months']:g} months' rent" if tm["deposit_months"] else "")
        kpi(k[4], "Lock-in / notice", f"{tm['lockin_months']:g} mo" if tm["lockin_months"] else "none found",
            f"Notice: you {fmt_notice(tm['notice_tenant_months'])} · landlord {fmt_notice(tm['notice_landlord_months'])}")
        st.write("")
        left, right = st.columns([1.5, 1], gap="large")
        with left:
            st.markdown("#### Flags, most serious first")
            show_ok = st.toggle("Show clauses that passed", value=False)
            for f in res["flags"]:
                if f["severity"] == "ok" and not show_ok:
                    continue
                where = f"Clause {f['clause_id']} · {f['clause_title']}" if f["clause_id"] else "Missing from agreement"
                near = (' <span class="cc-pill cc-low">near a threshold</span>' if f["near_threshold"] else "")
                st.markdown(f'<div class="cc-flag {f["severity"]}">{pill(f["severity"])}<span class="cc-small">{where}</span>{near}'
                            f'<div class="t">{f["title"]}</div><div class="f">{f["finding"]}</div>'
                            f'<div class="a">👉 {f["ask"]}</div></div>', unsafe_allow_html=True)
            if any(f["near_threshold"] for f in res["flags"]):
                st.caption("*Near a threshold*: the value sits close to a benchmark cut-off, so a small change "
                           "(e.g. 6 vs 6.5 months' deposit) would change the rating. Treat it as borderline.")
        with right:
            st.markdown("#### Risk by clause")
            sev = R.clause_severity(res["flags"])
            rows = [{"Clause": f"{c['id']}. {c['title'][:22]}", "Severity": sev.get(c["id"], "ok").capitalize().replace("Ok", "OK"),
                     "Level": max(0.15, R.SEV_ORDER[sev.get(c["id"], "ok")]), "order": i}
                    for i, c in enumerate(res["clauses"]) if c["type"] != "parties"]
            ch = alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Level:Q", scale=alt.Scale(domain=[0, 3]), axis=alt.Axis(values=[0, 1, 2, 3], title="Risk level (0 = OK, 3 = high)")),
                y=alt.Y("Clause:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
                color=alt.Color("Severity:N", scale=alt.Scale(domain=["OK", "Low", "Medium", "High"],
                                                              range=["#86efac", "#7dd3fc", "#fbbf24", "#ef4444"]), legend=alt.Legend(orient="bottom")),
                tooltip=["Clause", "Severity"],
            ).properties(height=max(260, 24 * len(rows)))
            st.altair_chart(ch, use_container_width=True)
            st.markdown("#### Key terms found")
            src = tm["sources"]
            terms_tbl = [
                ("Monthly rent", R.inr(tm["rent"]) if tm["rent"] else "—", src.get("rent", "")),
                ("Security deposit", R.inr(tm["deposit"]) if tm["deposit"] else "—", src.get("deposit", "")),
                ("Deposit refund", f"within {tm['refund_days']} days" if tm["refund_days"] else "not stated", ""),
                ("Lock-in", f"{tm['lockin_months']:g} months" if tm["lockin_months"] else "—", src.get("lockin", "")),
                ("Rent increase", f"{tm['escalation_pct']:g}%" if tm["escalation_pct"] is not None else "—", ""),
                ("Landlord visit notice", f"{tm['entry_notice_hours']:g} hours" if tm["entry_notice_hours"] else "—", ""),
            ]
            st.dataframe(pd.DataFrame(terms_tbl, columns=["Term", "Value", "Clause"]), hide_index=True, use_container_width=True)
            st.caption("Extracted by pattern rules. If a value says '—' or looks wrong, check the clause yourself.")

# ------------------------------------------------------------------ tab 3
def render_expl(e: dict):
    badges = []
    if e.get("basic_mode"):
        badges.append(pill("medium", "Rule-based (AI off)"))
    else:
        badges.append(pill("ok", "✓ AI did not soften any rule flag") if not e["raised_by_rules"]
                      else pill("high", f"⚠ AI rated {len(e['raised_by_rules'])} clause(s) too low, corrected"))
        badges.append(pill("ok", "✓ All amounts traced to the agreement") if not e["unverified"]
                      else pill("high", f"⚠ {len(e['unverified'])} figure(s) not in the agreement"))
        if e["bad_citations"]:
            badges.append(pill("high", f"⚠ cites missing clause {', '.join(e['bad_citations'])}"))
        badges.append(pill({"high": "ok", "medium": "medium", "low": "high"}[e["confidence"]], f"Confidence: {e['confidence']}"))
        badges.append(f'<span class="cc-small">Gemini · {status.get("model") or ""} · {e.get("lang", "English")}</span>')
    st.markdown("".join(badges), unsafe_allow_html=True)
    if e.get("leak_warning"):
        st.error("The AI output repeats text from a hidden instruction in the agreement. Rely on the rule flags in tab ②.")
    if e["raised_by_rules"]:
        st.warning("Gemini rated clause(s) " + ", ".join(e["raised_by_rules"]) + " as less risky than the rule checks. "
                   "The higher rule severity is shown below.")
    if e["unverified"]:
        st.warning("These figures in the explanation could not be found in the agreement: " + ", ".join(e["unverified"])
                   + ". Check them against the clause text.")
    st.markdown(f"**Overview.** {e['overview']}")
    a, b = st.columns(2)
    with a:
        st.markdown("**Top concerns**\n" + ("\n".join(f"- {x}" for x in e["top_concerns"]) or "- None"))
    with b:
        st.markdown("**Ask the landlord before signing**\n" + "\n".join(f"- {x}" for x in e["questions_for_landlord"]))
    if e["extra_observations"]:
        st.info("**AI-only observations** (not checked by rules):\n" + "\n".join(f"- {x}" for x in e["extra_observations"]))
    st.markdown("#### Clause by clause")
    for c in e["clauses"]:
        tag = " · raised by rules" if c["corrected"] else ""
        with st.expander(f"Clause {c['id']} · {c['title']} — {SEV_LABEL[c['severity']]}{tag}", expanded=c["severity"] == "high"):
            st.markdown(f"{pill(c['severity'])}", unsafe_allow_html=True)
            st.markdown(f"**In plain words:** {c['plain']}\n\n**For you:** {c['for_you']}")
            orig = next((x["text"] for x in res["clauses"] if x["id"] == c["id"]), "")
            st.caption("Original: " + (orig[:500] if not R.looks_like_injection(orig) else "[text addressed to AI tools]"))
    if e["not_explained"] and not e.get("basic_mode"):
        st.caption("Not explained by the AI: clause(s) " + ", ".join(e["not_explained"]) + ". See tab ② and the original text.")
    st.caption("ClauseClear is decision support, not legal advice. You remain responsible for what you sign.")
    md = (f"# ClauseClear checklist\n\nGenerated {datetime.now():%d %b %Y %H:%M} · {ss.source}\n\n"
          f"Score: {res['summary']['score']}/100 ({res['summary']['grade']})\n\n## Overview\n{e['overview']}\n\n## Top concerns\n"
          + "\n".join(f"- {x}" for x in e["top_concerns"]) + "\n\n## Ask the landlord\n"
          + "\n".join(f"- [ ] {x}" for x in e["questions_for_landlord"]) + "\n\n## Rule flags\n"
          + "\n".join(f"- **{SEV_LABEL[f['severity']]}** (Clause {f['clause_id'] or '-'}) {f['title']}: {f['finding']}"
                      for f in res["flags"] if f["severity"] != "ok")
          + "\n\n_Not legal advice._\n")
    st.download_button("⬇ Download checklist (.md)", md, "clauseclear_checklist.md", "text/markdown")


with t3:
    if not res:
        st.info("Load an agreement in tab ① first.")
    else:
        key = (res["hash"], ss.lang)
        st.markdown(f"Explaining **{ss.source}** in **{ss.lang}** ({len(res['clauses'])} clauses).")
        if st.button("✨ Explain this agreement" if AI_ON else "Show rule-based explanation", type="primary"):
            e = None
            if AI_ON:
                with st.spinner("Gemini is reading the agreement clause by clause…"):
                    try:
                        e = cached_explain(API_KEY, MODELS, json.dumps(payload, ensure_ascii=False), ss.lang)
                        ss.ai_calls += 1
                    except ai.AIError as err:
                        st.warning(f"AI unavailable: {err.message} Showing the rule-based version instead.")
                        if err.kind in ("key", "quota", "network"):
                            check_ai.clear()
            ss.expl = e or ai.fallback_explanation(payload, res)
            ss.expl_key = key
        if ss.expl:
            if ss.expl_key != key:
                st.warning("The agreement or language changed after this explanation was written. Click the button to refresh it.")
            render_expl(ss.expl)

# ------------------------------------------------------------------ tab 4
with t4:
    st.caption("Ask about the loaded agreement, e.g. *Can I leave after 4 months?* · *When do I get my deposit back?* · "
               "*What should I negotiate first?* ClauseClear is an AI assistant, not a lawyer or a person.")
    box = st.container(height=430)
    with box:
        if not ss.chat:
            st.markdown("👋 Hi, I'm **ClauseClear**, an AI assistant. I answer questions about the agreement you loaded, "
                        "citing its clauses. For legal disputes, I'll point you to a lawyer.")
        for msg in ss.chat:
            with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📜"):
                st.markdown(msg["content"])
    q = st.chat_input("Ask about this agreement…", max_chars=ai.MAX_INPUT_CHARS)
    if q:
        q = q.strip()
        ss.chat.append({"role": "user", "content": q})
        if not res:
            reply = "No agreement is loaded yet. Load a sample or paste yours in tab ① first."
        elif R.looks_like_injection(q):
            reply = ("I can't change my instructions or reveal them. I can only explain this rental agreement. "
                     "Try: *What happens to my deposit if I leave early?*")
        elif not AI_ON:
            top = [f for f in res["flags"] if f["severity"] in ("high", "medium")][:3]
            reply = (f"AI chat is off ({status['msg']}). From the rule checks, the main issues are: "
                     + "; ".join(f"{f['title']} (Clause {f['clause_id']})" for f in top) + ". See tab ②.")
        else:
            try:
                with st.spinner("Reading the clauses…"):
                    reply = get_client(API_KEY, MODELS).answer(q, payload, ss.chat[:-1], ss.lang)
                ss.ai_calls += 1
                bad = ai.unverified_figures(reply, payload)
                if bad:
                    reply += f"\n\n⚠️ *Checked by the app: these figures are not in the agreement: {', '.join(bad)}*"
                cites = ai.bad_citations(reply, payload)
                if cites:
                    reply += f"\n\n⚠️ *Checked by the app: the agreement has no clause {', '.join(cites)}.*"
            except ai.AIError as err:
                reply = f"Sorry, the AI is unavailable right now ({err.message}). The risk report in tab ② still works."
        ss.chat.append({"role": "assistant", "content": reply})
        st.rerun()
    if ss.chat and st.button("Clear chat"):
        ss.chat = []
        st.rerun()

# ------------------------------------------------------------------ tab 5
with t5:
    st.markdown("#### How ClauseClear works")
    st.markdown(
        "1. **Python reads the agreement**: splits it into numbered clauses, classifies each one, and extracts rent, "
        "deposit, lock-in, notice periods, rent increase, late fees and visit notice.\n"
        "2. **Python flags risks** against the benchmarks below and computes the tenant-friendliness score "
        "(starts at 100; each high, medium and low flag lowers it).\n"
        "3. **Gemini explains**: it receives the masked clauses plus the rule findings and writes plain-language "
        "explanations and questions for the landlord.\n"
        "4. **The app checks the AI**: it cannot rate a flagged clause lower than the rules, every amount it quotes "
        "must exist in the agreement, cited clause numbers must exist, and text aimed at AI tools is stripped out first."
    )
    st.markdown("#### Benchmarks used (tenant's view)")
    st.dataframe(pd.DataFrame(R.BENCHMARKS, columns=["Check", "Benchmark"]), hide_index=True, use_container_width=True)
    st.caption("Benchmarks are simplified market practice and references to the Model Tenancy Act, 2021, which applies "
               "only in states that have adopted it. They are a starting point for negotiation, not legal rules.")

calls_slot.caption(f"AI calls this session: {ss.ai_calls}/{ai.MAX_CALLS_PER_SESSION}")
