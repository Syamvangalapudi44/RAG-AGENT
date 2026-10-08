# RAG over a contradictory corpus

```
pip install -r requirements.txt
python run.py                      # full evaluation: 2 chunkers x {baseline, resolved} x 8 queries
python run.py --ask "What is the international travel per-diem?"   # one question
```
Outputs land in `results/` (`comparison.md`, `per_query.csv`, `results.json`, `chunk_size_distribution.png`).

## Using a real LLM (do this before you submit!)
Without configuration the project uses an *offline extractive generator* (so it runs anywhere). The task asks for
LLM-grounded generation, so set one of:
```
# OpenAI-compatible (OpenAI, your portal's LLM proxy, Groq, etc.)
export LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini LLM_API_KEY=sk-... [LLM_BASE_URL=https://.../v1]
# Ollama (local)
export LLM_PROVIDER=ollama LLM_MODEL=llama3
# Anthropic
export LLM_PROVIDER=anthropic LLM_MODEL=claude-haiku-4-5 LLM_API_KEY=...
python run.py
```
Windows PowerShell: `$env:LLM_PROVIDER="openai"` etc. Optional dense embeddings: `EMBED=st` (needs sentence-transformers).

## Files
| File | Purpose |
|---|---|
| `corpus/` | the 6 documents, byte-for-byte as given |
| `chunkers.py` | fixed-size sliding window **and** structure-aware (clause / table-row / Q&A / memo) chunkers |
| `embedder.py` | TF-IDF embeddings + NumPy cosine index (optional sentence-transformers / OpenAI embeddings) |
| `meta.py` | ingestion-time metadata: effective dates, "supersedes" chain, memo -> clause amendment links |
| `resolver.py` | conflict resolution: as-of date, drop superseded docs, redact amended clauses, provenance notes |
| `generator.py` | grounded generation (LLM or offline extractive), citations, abstention |
| `pipeline.py` | ties it together; `mode="baseline"` = naive top-3, `mode="resolved"` = metadata resolution + abstention |
| `evaluate.py` | the 8 queries, gold checks, error taxonomy, metrics, tables, plot |
| `run.py` | entry point |
| `DECISIONS.md`, `ERROR_ANALYSIS.md`, `LIMITATIONS.md`, `WRITEUP.md` | the written deliverables |

---
# Q2 - Agent with unreliable tools
```
python run_agent.py                                  # 10-query eval + seeded stress test
python run_agent.py --ask "What is 18% GST on INR 12,500?"
python run_agent.py --ask "What is the total per-diem allowance in INR for a 7-day international trip?" --fault lookup_policy:1:wrong
```
Same `LLM_PROVIDER / LLM_MODEL / LLM_API_KEY` variables as Q1. **Unset = scripted MockLLM (not a real model)** - re-run with a real model before submitting.
Outputs: `outputs/agent_eval.csv`, `outputs/stress_eval.csv`, `outputs/agent_run_info.txt`, `traces/*.json|txt` (one per eval query).

| File | Purpose |
|---|---|
| `tools.py` | calculator, `lookup_policy` (the Q1 retriever), `convert_currency`; `FaultPlan`/`ToolBox` inject timeout / error / plausible-wrong value |
| `llm.py` | LLM client (OpenAI-compatible / Ollama / Anthropic) + scripted MockLLM |
| `agent.py` | ReAct loop, termination caps, retries, verification-triggered re-run |
| `verifier.py` | separate verification pass (programmatic checks + its own auditor LLM call) |
| `run_agent.py` | evaluation, ablation (verification off), stress test, traces |
