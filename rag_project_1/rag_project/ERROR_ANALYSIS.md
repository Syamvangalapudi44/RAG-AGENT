# ERROR ANALYSIS
Source: `results/per_query.csv`. Generator for these numbers: **offline extractive stand-in** (see LIMITATIONS #1).
Labels come from rule-based classification in `evaluate.py`, then re-read by hand.

## Summary (✓ = correct)
| Q | fixed / baseline | fixed / resolved | structure / baseline | structure / resolved |
|---|---|---|---|---|
| 1 intl per-diem | conflict | ✓ | conflict | ✓ |
| 2 equipment limit | fragment | fragment | ✓ | ✓ |
| 3 contractor laptop | negation | negation | negation | negation |
| 4 Lead notice | fragment | fragment | ✓ | ✓ |
| 5 receipt threshold | conflict | ✓ | conflict | ✓ |
| 6 per-diem June 2019 | conflict | ✓ | conflict | ✓ |
| 7 business class | ✓ | ✓ | ✓ | ✓ |
| 8 parental leave | fabricated | ✓ (abstained) | fabricated | ✓ (abstained) |

## Failures before the fix
* **Q1, Q5 - conflict.** 2019 and 2024 clauses are textually near-identical, so they score almost the same; the naive top-3 contains both and the answer prints both numbers (e.g. 9,000 *and* 6,500) with no hint which governs. Ties break by corpus order, i.e. arbitrarily. Silent-pick/no-pick failure.
* **Q6 - conflict (reverse direction).** The *newer* document is the wrong one here: June 2019 predates the 2024 policy. Baseline mixes 2019 and 2024 values.
* **Q2 - fragment (fixed only).** The 200-char window ends mid-number ("...raised to INR"), so "40,000" is in the next chunk. Structure chunking keeps the memo whole. Note the naive pipeline got Q2 right with structure chunks only because the memo happens to share the word "limit" with the query; the 25,000 clause (which says "shall not exceed") ranked lower. That is luck, not design.
* **Q4 - fragment (fixed only).** The window splits the table; the best-scoring text is the header row, the "Lead / Manager | 90 days" row is in a different, lower-ranked chunk.
* **Q8 - fabricated.** Naive pipeline always answers; it returned an unrelated "company ..." sentence for a topic the corpus never mentions.

## Still wrong after the fix
* **Q3 - negation (all four configs).** Asked "I am a contractor. Will the company buy me a laptop?", the stand-in generator returns clause 5.1 ("The company provides a laptop to every full-time employee"), because "company" and "laptop" outweigh "contractor" lexically. The correct clause (5.3, contractors not covered) is in the retrieved set - this is a **generation** failure of the extractive stand-in, not a retrieval or conflict failure. A real LLM should read 5.1 and 5.3 together; this must be re-measured with an LLM (not done here).
* **Q2 with fixed chunking - fragment**, unchanged by the resolver: resolution cannot repair a truncated number. Fix: structure chunking (already adopted) or larger window with overlap > number length.
* **Q4 with fixed chunking - fragment**, same reason.

## What the fix changed
Resolution (metadata + abstention) fixed Q1, Q5, Q6, Q8 for both chunkers. It did nothing for Q3 (not a conflict problem) or for fixed-chunk fragments (Q2, Q4).

---
# Q2 error analysis (mock planner; see `outputs/agent_eval.csv`, `outputs/stress_eval.csv`)
* **Without verification** 3 of the 5 fault runs are silently wrong (Q6 wrong calculator, Q8 wrong lookup -> 7,000 instead of 63,000, Q9 wrong conversion). Q8 is the instructive one: the calculator was right, the *input* was wrong.
* **With verification** all 10 correct; 0 silent errors in 90 stress runs, but 8 honest failures: 4 were refusals after a tool stayed down through all retries (correct behaviour, counted as not-correct), 3 were answers tagged `[UNVERIFIED]` because several wrong-value faults in a row exhausted the 2 verification rounds, and 1 hit the step cap (`INCOMPLETE`). Each of these is visible to the user as a failure; none presents a wrong number as fact.
* **Bug found by testing:** honest refusals were first false-flagged by the number-support check ("3 attempts" is not in any tool output). Fixed by exempting declared refusals. This is a false-positive class a real LLM auditor will have too.
* **Not exercised:** the stale-but-genuine retrieval, a wrong exchange *rate*, malformed/looping LLM output (the caps exist but no test triggers them).
