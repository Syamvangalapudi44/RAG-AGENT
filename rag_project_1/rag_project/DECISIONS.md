# DECISIONS

| # | Decision | Alternatives considered | Why |
|---|---|---|---|
| 1 | Two chunkers: **fixed** (200-char window, 40 overlap) vs **structure-aware** (one chunk per numbered clause / table row / Q&A pair / memo, each prefixed with document title + effective date) | Sentence chunking; recursive splitter | Two implementations that fail differently: the fixed window cuts clauses and tables mid-way (it cut "INR 40,000" in half); structure-aware chunks keep each fact whole. |
| 2 | Table rows become self-describing chunks (`Band: 4 \| Role level: Lead / Manager \| Notice period: 90 days`) | Keep table as one chunk | A row without its column headers is meaningless; a whole-table chunk makes the answer a needle in a small haystack. |
| 3 | Embeddings: TF-IDF vectors + NumPy cosine (default) with optional MiniLM / OpenAI embeddings | Dense model only | Runs offline with zero downloads; the corpus has 22 chunks so a vector DB is overkill. Trade-off: purely lexical, no synonym matching (e.g. "approves" vs "approval"). |
| 4 | **Conflict design = effective-date metadata extracted at ingestion + provenance disclosure** | Recency ordering by date only; ask the user to choose every time; refuse on any conflict | See below. |
| 5 | Version chain from the phrase "Supersedes the policy dated ..."; amendment link from "clause 5.2 of the Equipment and Asset Policy" (regex, no LLM) | Ask an LLM to extract | Deterministic and auditable. Weakness: breaks if wording changes (see LIMITATIONS). |
| 6 | As-of date parsed from the query ("June 2019"); default = today | Always use latest | Query 6 *requires* the superseded document; "always newest" would answer it wrongly. |
| 7 | Amended clause is **redacted** from the original chunk and the memo chunk takes its place | Boost the memo's score | The memo never uses the words of the clause ("shall not exceed"), so lexical/semantic ranking can't be trusted to link them. The explicit clause reference can. |
| 8 | Every answer carries a `NOTE:` line saying which version governed and what was set aside | Silent resolution | The assignment forbids silently picking one. |
| 9 | If two different in-force versions of the same document family remain and metadata can't order them, return "sources conflict" and list both files | Pick latest by filename | Never triggered in this corpus; the code path is **untested** (see LIMITATIONS). |
| 10 | Abstain when top cosine < 0.15 **or** < 30% of the query's idf-weighted terms appear in the evidence (out-of-vocabulary query terms count at max weight) | Cosine threshold only | Cosine alone could not separate Q8 (parental leave: top score 0.25) from Q3 (legit: 0.32). Coverage separates them (Q8 = 0.14 vs lowest answerable query = 0.44; threshold 0.30 sits in that gap and was chosen after seeing these numbers - see LIMITATIONS). |

## Why metadata + disclosure rather than the alternatives
* *Recency ordering alone* fails Q6 (needs the old policy) and cannot link the memo to clause 5.2.
* *Always asking the user* is safe but pushes work onto the user for the 3 queries where the corpus itself says which version governs (a supersedes line, an explicit clause reference). Wasteful when the answer is determinable.
* *Refusing on conflict* would have refused Q1, Q2, Q5, Q6 - four of eight - although the corpus resolves each.
* Chosen design: resolve **only where the documents themselves give the ordering**, state what was done, and fall back to "surface both" when they don't.

---
# Q2 decisions
| # | Decision | Why / alternative rejected |
|---|---|---|
| A1 | Tools: calculator (safe AST evaluator, no `eval`), `lookup_policy` (Q1 structure chunks + conflict resolution, top-2), `convert_currency` (fixed 83 INR/USD) | The 3rd tool has an independent recomputation path, which is what makes wrong-value detection testable. |
| A2 | Faults drawn per (seed, tool, n-th call) | Verification on/off then see identical faults -> paired ablation; retries and reruns draw fresh outcomes. |
| A3 | Three fault kinds: timeout (daemon thread, 0.15 s deadline), error (HTTP-style text or truncated JSON, caught by schema validation), wrong (well-formed, same shape, subtly different number) | Matches the brief; "wrong" mutates the *value*, never the shape, so nothing signals failure. |
| A4 | Harness retries transient failures twice, then tells the model `ERROR: ... unavailable`; the model must refuse, not guess | Retrying is cheap and deterministic; guessing after outage is the dangerous behaviour. |
| A5 | Verifier = Layer A (independent recomputation / source grounding / answer-support) + Layer B (own LLM call, own system prompt, no access to the agent's thoughts) | A pure-LLM verifier cannot be trusted on arithmetic; a purely programmatic one cannot judge consistency. |
| A6 | On failure: re-run the *earliest flagged step*, discard everything after it, continue | Later steps may have consumed the bad value (Q8: 7*1000). Re-running only the final answer would keep the poisoned input. |
| A7 | Max 2 verification rounds, then return `[UNVERIFIED]` | Bounded cost; an unverified-but-flagged answer beats an infinite loop or a silent pass. |
| A8 | Numeric answer-support check ignores declared refusals | Otherwise honest refusals are false-flagged (found in testing). |
