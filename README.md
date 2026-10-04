# ClauseClear: Rental Agreement Explainer

AI for Managers end-term project, Use Case #19 (Legal-clause explainer for a specific contract type).
Built by **Harsh Agarwal** (Roll No. 065078).

ClauseClear reads an Indian 11-month residential rental (leave-and-license) agreement from the **tenant's** side:

1. **Python reads and flags** – splits the agreement into clauses, extracts rent, deposit, lock-in, notice periods,
   rent increase, late fees and visit notice, and flags risks against simple benchmarks (tenant-friendliness score 0–100).
2. **Gemini explains** – plain-language explanation of every clause in English or Hindi, top concerns and questions
   to ask the landlord, plus a chat that answers questions citing clause numbers.
3. **The app checks the AI** – Gemini cannot rate a flagged clause lower than the rules, every amount it quotes must
   exist in the agreement, cited clauses must exist, and text addressed to AI tools inside the agreement is removed.

Personal details (names after honorifics, phone, e-mail, PAN, Aadhaar, account numbers) are masked before any API call.
If Gemini is unavailable, the app falls back to a rule-based mode. **Not legal advice.**

## Run locally
```bash
pip install -r requirements.txt
mkdir -p .streamlit && echo 'GEMINI_API_KEY = "your-key"' > .streamlit/secrets.toml
streamlit run app.py
python -m pytest -q tests      # 13 tests, no network needed
```
Sample agreements in `data/samples/` are fictional.
