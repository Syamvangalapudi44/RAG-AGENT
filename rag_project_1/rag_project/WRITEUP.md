# WRITEUP

**Tools/models used.** Python 3.12, NumPy, matplotlib. Embeddings: TF-IDF (NumPy cosine). Generation for the numbers below: **offline extractive generator** (no LLM was reachable when these were produced).
> **[EDIT BEFORE SUBMITTING]** Re-run `python run.py` with your LLM (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`) and replace the numbers in this file with the new `results/comparison.md`. State the exact model name here: `__________`.

## Results (8 queries, `results/comparison.md`)
| Chunking | Pipeline | Accuracy | P@3 | Chunks | Mean size (chars) | Abstention |
|---|---|---|---|---|---|---|
| fixed (200/40) | baseline | 1/8 (12%) | 0.29 | 16 | 169 | 0% |
| fixed (200/40) | + conflict resolution | 5/8 (62%) | 0.33 | 16 | 169 | 12% |
| structure-aware | baseline | 3/8 (38%) | 0.43 | 22 | 135 | 0% |
| structure-aware | + conflict resolution | **7/8 (88%)** | 0.38 | 22 | 135 | 12% |

Chunk sizes: fixed 65-200 chars (median 200); structure 88-289 (median 126). Plot: `results/chunk_size_distribution.png`.

## Which strategy I would deploy
Structure-aware chunking with the resolution layer. Measured: 7/8 vs 5/8 for fixed windows with the same resolver, and 3/8 vs 1/8 without it. The difference comes from fragments: the fixed window truncated "INR 40,000" (Q2) and separated the Lead row from its table (Q4); structure chunks did neither. Its P@3 is also higher (0.38 vs 0.33 with resolution; 0.43 vs 0.29 without). Resolution slightly lowers P@3 for structure chunks (0.43 -> 0.38) because it removes superseded chunks that were "relevant" by text but not in force; I accept that trade, since precision against raw text is not the goal when half the text is stale.

## Conflict-resolution design (Step 4)
At ingestion, regex extracts each document's effective date, the "Supersedes ... dated <date>" link, and the memo's "clause 5.2 of the Equipment and Asset Policy" link. At query time: (1) parse an as-of date from the question (default today); (2) drop documents not in force on that date; (3) replace an amended clause with the amending memo; (4) attach a `NOTE:` line stating which version governed and what was set aside; (5) if two in-force versions remain unordered, say so and list both rather than pick. I chose this over pure recency (fails Q6, can't link the memo), always-ask-the-user (pushes work to the user where the documents already say which governs), and refuse-on-conflict (would refuse 4 of 8 queries). Before/after: 1/8 -> 5/8 (fixed), 3/8 -> 7/8 (structure). Q1, Q5, Q6 (conflict) and Q8 (fabrication) flipped to correct in both.

## Queries still wrong after the fix
* **Q3 (all configs) - negation.** The stand-in generator picks "The company provides a laptop..." over the contractor exclusion. The right clause is retrieved, so this is generation. With more time: run with a real LLM and measure; add a prompt rule to check exclusions/eligibility; add an eval for "role-conditional" questions.
* **Q2 and Q4 under fixed chunking - fragment.** Not fixable by the resolver; solved by choosing structure chunking.

## What this evaluation cannot tell you
Eight queries means one query = 12.5 points; 7/8 has a ~53-98% 95% interval, so 7/8 vs 5/8 is not a reliable ranking by itself. The queries were designed around known failure modes, so accuracy measures "does it handle these traps", not general quality. Thresholds and stop-words were tuned after seeing these queries (no held-out set), so the 88% is optimistic. The numbers are from an extractive stand-in, not an LLM. P@3 is capped by the corpus (1-2 relevant chunks per query).

## Where I used the AI assistant (and one place I did not accept its output)
> **[EDIT - write this truthfully in your own words.]** Draft for you to correct: I used Claude (claude.ai chat) to scaffold the project - chunkers, TF-IDF index, metadata extractor, resolver, evaluation harness - and to draft the markdown files. I ran the code, read per-query outputs, and checked the numbers in this file against `results/per_query.csv`.
> One place I did *not* accept its output: **[fill in a real example]**. Candidates from the build: (a) the first abstention gate used a retrieval-score threshold only, which could not separate Q8 (0.25-0.31) from the legitimate Q3 (0.23-0.32), so it was replaced with an idf-weighted query-coverage rule; (b) an early version attached the 5.2-amendment note to every answer (even notice-period questions) and had to be filtered to notes about evidence actually cited; (c) a stop-word edit accidentally removed "per" from the vocabulary and silently broke Q6 until the results were checked. Only list the ones you actually reviewed and agree with.

---
# Q2 - Agent with unreliable tools: verification and self-correction

**Model/tools.** Python 3.12, NumPy; Q1 retriever reused as `lookup_policy`. **The results below were produced with a scripted `MockLLM` (hand-written planner + an auditor stub that echoes the programmatic findings), because no LLM was reachable when I built this.** The loop, fault injection, verifier checks and trace/eval machinery are real and exercised; tool *selection* and the auditor's free-form judgement are not. The real-LLM path (OpenAI-compatible / Ollama / Anthropic) is written but unrun here.
> **[EDIT BEFORE SUBMITTING]** Run `python run_agent.py` with your model, paste the new numbers below, and name the model: `__________`. A real model may fail differently (bad JSON, wrong tool choice) - that is exactly what the caps below are for.

## Results (`outputs/agent_eval.csv`, 10 queries, mock planner)
| | n | correct with verification | correct without (same faults) |
|---|---|---|---|
| simple (Q1-2) | 2 | 2 | 2 |
| multi-step, >=3 calls (Q3-5) | 3 | 3 | 3 |
| injected faults (Q6-10) | 5 | 5 | 2 |
| **total** | 10 | **10** | **7** |

Verification fired in 3 runs (Q6 wrong calculator, Q8 wrong lookup, Q9 wrong conversion); in all 3 it **helped** (wrong -> right) and never hurt. Q7 (timeout + error + timeout) was recovered by retries alone; Q10 (lookup permanently down) ended in an honest refusal, which is the correct outcome.

Stress test (`outputs/stress_eval.csv`): 9 questions x 10 seeds = 90 runs per arm, random faults (timeout 10%, error 10%, plausible-wrong 15% per call), same fault schedule with verification on vs off:
| | correct | honest failure (refused / flagged UNVERIFIED) | **silently wrong** |
|---|---|---|---|
| verification ON | 82 | 8 | **0** |
| verification OFF | 57 | 2 | **31** |

## Loop termination
The agent ends itself by emitting `finish`; in every eval run `terminated_by` is `finish` (2-5 LLM steps), never a cap. Hard stops exist for when the model misbehaves: `step_cap` (10 LLM steps), `loop_detected` (same action+input 4x), `malformed_output` (3 unparseable replies; each bad reply is fed back as an error first). Each returns an explicit `INCOMPLETE: ...` answer with the partial tool results rather than a guess, and records the reason in the trace. Verification gets 2 re-run rounds; if it still fails the answer is returned tagged `[UNVERIFIED: ...]`. In the stress test, `step_cap` occurred once and `finish_unverified` 3 times (of 90), all under several consecutive faults on the same question.

## How a wrong-but-plausible tool result is detected - and how reliably
The verifier is a separate step (own system prompt, sees only question + tool calls/outputs + draft + automated findings, not the agent's thoughts). Layer A is programmatic and does **not** trust the tools: (1) arithmetic - it re-evaluates every calculator expression / conversion itself; (2) grounding - every lookup's excerpt must appear verbatim in the cited file on disk, the lookup text may contain no number absent from that excerpt, and a superseded source is flagged; (3) support - every number in the answer must come from the question or a tool output and every cited file must have been returned by a lookup. Layer B is an auditor LLM call that can flag what Layer A cannot. Any flag -> fail.

Honest reliability: on *this* fault model it caught 3/3 injected wrong values and the stress run had 0 silent errors in 90 runs - but that is because the injected corruptions are the kind that can be checked independently (arithmetic is recomputable; a mutated number is not in the source file). It would **not** catch: a wrong value that is internally consistent and present in the source (e.g. the lookup returns the 2019 clause labelled 2019 for a "current" question - only the superseded-file check helps, and only because Q1 metadata exists); a wrong conversion *rate* (verifier and tool share the constant); a wrong-but-valid retrieval of an irrelevant clause; or any tool whose output has no independent ground truth. Layer B in the mock only echoes Layer A, so its real value is unmeasured.

## One trace where verification changed the answer (`traces/q08_fault-wrong-lookup.txt`)
Question: total per-diem for a 7-day international trip. Fault: first `lookup_policy` call returns a well-formed but tampered result ("INR 1,000" instead of 9,000; same file, same score, no error).
1. Step 1 lookup -> "3.2 International travel per-diem is INR 1,000 per day." Step 2 calculator `7*1000` -> 7000 (arithmetically correct on wrong input).
2. Draft: "INR 7,000". A try/except agent stops here. The calculator check passes (7*1000 = 7000) - the arithmetic is *consistent*, which is why checking the calculator alone is not enough.
3. Verifier Layer A grounding check: the lookup text contains the number 1000 which is absent from its own source excerpt ("...INR 9,000 per day.") -> fail, issue attached to step 1.
4. Agent discards step 1's downstream steps, re-runs lookup (call #2, fault-free) -> 9,000, re-plans: calculator `7*9000` -> 63000.
5. Re-verification passes. **Final: INR 63,000** (draft was INR 7,000). Events recorded: `verification_rerun step=1, discarded_later_steps=1`. Without verification the same fault plan gives the wrong 7,000 (`correct_without_verification=False`).
Q6 (calculator returns 54,360 for `6*9000 + 2*3200`, truth 60,400) and Q9 (conversion off by ~4%) follow the same pattern.

## Failure mode I did not handle
**Wrong-but-consistent data from a ground-truth-less tool**, and relatedly, **correlated failures** (verifier and tool sharing one wrong source, e.g. the same stale exchange-rate table). In production this means a confidently wrong answer carrying a "verified" stamp, which is worse than no stamp because users stop double-checking. Cost to fix: independent second sources per tool (a second retriever/rate feed), disagreement escalation to a human, and a measured false-negative rate on real faults. Also unhandled: latency/cost (a re-run roughly doubles calls), partial-failure answers (one leg of a multi-leg trip fails), and prompt-injection via retrieved text.

## Where I used the AI assistant, and one suggestion I rejected
> **[EDIT - write truthfully.]** Draft: I used Claude (claude.ai chat) to scaffold the agent loop, fault injector, verifier, MockLLM and eval scripts, then ran and read the traces myself. One thing I rejected: **[fill in]**. Real candidates from the build, only keep ones you agree with: (a) having the verifier simply ask the same model "is this right?" - rejected because an LLM cannot reliably recompute arithmetic or notice a tampered number, so Layer A recomputes independently and the LLM only audits what remains; (b) only wrapping tools in try/except for timeouts/errors - rejected as it never notices plausible-wrong values; (c) a first version flagged honest refusals ("I could not complete...") as unsupported because their error text contained "3 attempts" - fixed by exempting declared refusals from the number-support check, not by loosening the check.
