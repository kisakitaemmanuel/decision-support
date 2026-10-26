# offline_ucg - offline clinical decision support from the Uganda Clinical Guidelines 2023

Free-text patient note -> red-flag rules -> hybrid retrieval (FTS5 + FAISS) -> ranked conditions -> medicine/dose/
contraindication review -> referral advice -> sourced report -> audit record. Runs with no internet.

```
offline_ucg/
├── app.py  cli.py  config.py  db.py  schemas.py
├── database/   schema.sql  ucg.db  ucg_vectors.npz  (ucg.faiss when faiss is installed)
├── etl/        run_etl.py  chunker.py  drugs.py  extract_pdf.py (PDF -> source DB)
├── retrieval/  sqlite_search.py  semantic_search.py  hybrid.py  embedders.py
├── rules/      engine.py  red_flags.py  contraindications.py  dosing.py  referral.py  signoff.json
├── agents/     base.py (tool registry + loop)  llm.py  patient_agent.py  search_agent.py
│               clinical_agent.py  medication_agent.py  explanation_agent.py  orchestrator.py
├── audit/      logger.py  decisions.db (created at first run)
├── models/     clinical_model.gguf (you supply)  embeddings/
├── ui/desktop/ (PySide6)      ui/android_flutter/ (Flutter)
└── tests/test_system.py
```

## Run
    pip install -r requirements.txt
    python -m etl.run_etl --source ../ucg/ucg_2023.db        # or --pdf UCG-2023-Publication-Final-PDF.pdf
    python cli.py ask "3 year old, 14 kg, fever 39 C, convulsion, not able to drink" --trace
    uvicorn app:app --host 0.0.0.0 --port 8000               # API + docs at /docs
    python -m ui.desktop.main                                # desktop (add --api http://host:8000 for a shared server)
    python -m unittest discover -s tests -v

## How it is built to be debuggable
- `python cli.py health` shows which parts are really active (embedder, FAISS or numpy, LLM or deterministic).
- Every result carries `trace` (per-step timings), `retrieval.candidates[].ranks` (rank from each retriever) and the tool-call trace.
- Failures degrade, never crash: semantic search off -> FTS only (reported in `degraded`); LLM off or crashing -> deterministic ranking and template report.
- Rules are rows in SQLite (`red_flag_rules`, `contraindication_rules`, `dose_rules`), JSON conditions documented in `rules/engine.py`; unknown keys raise at ETL validation.
- Logs: console + `logs/ucg.log`; `UCG_LOG_LEVEL=DEBUG` or `python cli.py -v ...`.
- Settings: `config.py`, every value overridable as `UCG_<NAME>`.

## Safety design (read this)
1. Red flags run first, in plain rules, before any model.
2. The LLM never produces doses, drugs or thresholds. Doses come only from `dose_rules`; drugs only from the guideline text (ETL); warnings only from `contraindication_rules`.
3. The LLM may nominate conditions (must exist in the DB, bounded bonus `llm_rank_bonus`) and write a short summary, which is discarded if it contains any number or drug not present in the structured report.
4. Rules not signed off by a clinician show `unreviewed` in every output. To sign off a rule add `"RULE-ID": "reviewer, date"` to `rules/signoff.json` and re-run the ETL; set `UCG_REQUIRE_REVIEWED_RULES=1` to ignore unreviewed rules entirely.
5. Unknown inputs never fire a rule; they are listed as "information still needed".
6. Every decision is stored in `audit/decisions.db` with rules fingerprint, embedder and LLM versions; clinicians can mark agree/disagree.
7. This is decision support built from an automatically extracted PDF. It is not a substitute for clinical judgement and has not been clinically validated.
