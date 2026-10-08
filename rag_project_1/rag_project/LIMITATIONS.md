# LIMITATIONS

1. **Generator.** The reported numbers were produced with the *offline extractive generator*, not an LLM (no network/API key in the environment where the project was built). It returns the best-matching clause(s) verbatim. It is a crude stand-in: it causes the Q3 failure and makes "accuracy" partly a statement about that stand-in. **Re-run `python run.py` with `LLM_PROVIDER` set and update the numbers** in WRITEUP.md; the LLM code path (OpenAI-compatible / Ollama / Anthropic via `urllib`) is written but was not executed here.
2. **Eight queries.** One query = 12.5 percentage points. 7/8 vs 5/8 vs 3/8 are not statistically distinguishable in any rigorous sense; there are no confidence intervals worth quoting (a 7/8 result has a 95% interval of roughly 53-98%). The queries also map almost 1:1 to the failure modes we then engineered fixes for.
3. **Tuned on the test set.** The question-filler stop-words (`need keep amount current taken tell give get`) and the abstention thresholds (cosine 0.15, coverage 0.30) were set after seeing these eight queries. The abstention gap (Q8 = 0.14 vs weakest answerable = 0.44) is wide here, but a 9th query could land inside it. No held-out set exists.
4. **Precision@3 is low (0.29-0.43) by construction**: with 1-2 relevant chunks and k=3, even a perfect retriever can't exceed 0.33-0.67 on most queries. Read it as relative, not absolute. Relevance is judged by a regex over the gold evidence, not by humans. Q8 is excluded (nothing is relevant).
5. **Metadata extraction is regex.** It relies on the exact phrases "Supersedes the policy dated <date>" and "clause N of the <Title>". A memo saying "we are bumping the home-office cap" with no clause number would not be linked. An LLM extractor with human review would be needed at scale.
6. **Conflict-surfacing path is untested.** The "two in-force versions, no ordering metadata" branch never fires on this corpus and has no test.
7. **As-of parsing is naive** (month+year, or "in/for/of <year>"). "Last year", "before the 2024 policy", etc. fall back to today's date.
8. **Lexical embeddings** cannot match synonyms or paraphrase; they happen to suffice because the queries reuse corpus vocabulary. Dense embeddings were not evaluated.
9. **Citations** in LLM mode are parsed from `[file.md]` tags the model writes, falling back to all retrieved files if it writes none - which could over-credit citations.
10. **Single run, deterministic** (temperature 0 / no randomness); variance across LLMs/prompts is unmeasured.

---
# Q2 limitations
1. **Mock LLM.** Planner is hand-written for these 10 questions and the auditor stub echoes Layer A. The results demonstrate the harness, not a model's behaviour. Re-run with a real LLM; expect worse tool selection, malformed JSON and different verifier behaviour.
2. **The verifier is matched to the fault injector.** Wrong values are mutations that independent recomputation / source comparison can catch. Real wrong-but-plausible failures (stale data, wrong source, shared bad constant) are not covered; 0 silent errors is a statement about this fault model only.
3. **Small, partly correlated samples.** 10 eval queries; the stress test is 90 runs per arm over only 9 distinct questions, so effective sample size is smaller than 90.
4. **Eval was written by the same person as the code**, and the 10 questions were fixed before tuning the verifier; the support-check exemption for refusals was added after seeing Q10.
5. Timeouts are simulated (a sleeping thread), not real network behaviour; threads are abandoned, not cancelled.
6. Retriever is the Q1 TF-IDF system and inherits its limits (see Q1 LIMITATIONS).
7. Fixed exchange rate (83 INR/USD) is a demo constant, not a live feed.
