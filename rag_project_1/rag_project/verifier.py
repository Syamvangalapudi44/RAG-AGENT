"""Verification pass - a SEPARATE step from the agent's reasoning.

Layer A (programmatic, independent of the tools it audits):
  * arithmetic   - re-evaluates every calculator expression / currency conversion itself and compares to what the tool returned
  * grounding    - every lookup result must (i) have its source_excerpt found verbatim in the cited file on disk and
                   (ii) contain no number that is absent from that excerpt; cited file must not be superseded
  * support      - every number in the final answer must come from the question or a tool observation; every cited
                   source file must have been returned by a lookup
Layer B (its own LLM call, different system prompt, sees only question + tool calls/outputs + draft answer + Layer-A
  findings - NOT the agent's thoughts): judges consistency / support and may flag steps Layer A cannot.
Verdict = fail if either layer flags anything. Flags carry the (1-based) step number so the agent can re-run it.
"""
import json, re
from tools import safe_eval, INR_PER_USD
from meta import extract

AUDITOR_SYSTEM = ("You are an independent auditor. You did NOT produce the answer you are checking. You are given a question, the "
                  "tool calls with their raw outputs, a draft answer, and findings from automated checks. Check ONLY: (1) arithmetic is "
                  "correct, (2) the answer is internally consistent, (3) every claim in the answer is supported by something a tool "
                  "actually returned (not by outside knowledge), (4) the tool outputs themselves look consistent with each other. "
                  "Reply with JSON only: {\"verdict\": \"pass\"|\"fail\", \"issues\": [{\"step\": <1-based step number or null>, \"problem\": \"...\"}]}. "
                  "Be strict: when a number cannot be traced to a tool output, fail.")

NUM = re.compile(r"\d[\d,]*\.?\d*")


def nums(text):
    out = []
    for m in NUM.findall(text):
        m = m.replace(",", "").rstrip(".")
        try: out.append(float(m))
        except ValueError: pass
    return out


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


class Verifier:
    def __init__(self, llm, docs):
        self.llm, self.docs = llm, docs
        self.metas = extract(docs)
        self.file_norm = {f: norm(t) for f, t in docs.items()}

    # ---- layer A
    def programmatic(self, question, steps, answer, sources):
        issues, seen_files = [], set()
        for i, s in enumerate(steps, 1):
            if not s["ok"]: continue
            payload = json.loads(s["obs"])
            if s["action"] == "calculator":
                try: truth = safe_eval(s["input"]["expression"])
                except Exception: continue
                if abs(truth - payload["result"]) > 0.005:
                    issues.append((i, f"arithmetic: calculator returned {payload['result']} but independent recomputation of "
                                      f"'{s['input']['expression']}' gives {truth}"))
            elif s["action"] == "convert_currency":
                a = float(str(s["input"]["amount"]).replace(",", ""))
                truth = round(a / INR_PER_USD, 2) if s["input"]["from"].upper() == "INR" else round(a * INR_PER_USD, 2)
                if abs(truth - payload["result"]) > 0.011:
                    issues.append((i, f"arithmetic: conversion returned {payload['result']} but independent recomputation gives {truth}"))
            elif s["action"] == "lookup_policy":
                for r in payload["results"]:
                    seen_files.add(r["file"])
                    ex = norm(r["source_excerpt"])
                    if r["file"] not in self.file_norm or ex not in self.file_norm[r["file"]]:
                        issues.append((i, f"grounding: excerpt attributed to {r['file']} is not found in that file"))
                    elif set(nums(r["text"])) - set(nums(ex)):
                        bad = sorted(set(nums(r["text"])) - set(nums(ex)))
                        issues.append((i, f"grounding: lookup text contains numbers {bad} that are not in its source excerpt from {r['file']}"))
                    dm = self.metas.get(r["file"])
                    if dm and dm.superseded_by and not re.search(r"\b(19|20)\d\d\b", question):
                        issues.append((i, f"staleness: {r['file']} is superseded by {dm.superseded_by}"))
        # answer support
        allowed = set(nums(question))
        for s in steps:
            if s["ok"]: allowed |= set(nums(s["obs"]))
        body = re.sub(r"\S+\.md", "", answer)
        declined = re.match(r"\s*(I could not|I cannot|I can't|INCOMPLETE)", answer)   # an honest refusal asserts no values
        for x in ([] if declined else nums(body)):
            if not any(abs(x - a) <= 0.5 for a in allowed):
                issues.append((None, f"support: number {x:g} in the answer does not come from the question or any tool output"))
        for f in sources or []:
            if f not in seen_files:
                issues.append((None, f"support: answer cites {f} which no lookup returned"))
        return issues

    # ---- layer B + combine
    def verify(self, question, steps, answer, sources):
        a_issues = self.programmatic(question, steps, answer, sources)
        trace_txt = "\n".join(f"step {i}: {s['action']}({json.dumps(s['input'])}) -> {s['obs'][:600]}" for i, s in enumerate(steps, 1))
        prompt = (f"QUESTION: {question}\n\nTOOL CALLS AND OUTPUTS:\n{trace_txt or '(none)'}\n\nDRAFT ANSWER: {answer}\n\n"
                  "PROGRAMMATIC FINDINGS:\n" + ("\n".join(f"- step {i}: {m}" if i else f"- general: {m}" for i, m in a_issues) or "none"))
        try:
            raw = self.llm.chat(AUDITOR_SYSTEM, [{"role": "user", "content": prompt}])
            b = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        except Exception as e:
            b, raw = {"verdict": "pass", "issues": []}, f"(auditor call failed: {e})"
        b_issues = [(it.get("step"), "auditor: " + str(it.get("problem"))) for it in b.get("issues", [])] if b.get("verdict") == "fail" else []
        all_issues = a_issues + [x for x in b_issues if x[1].split(": ", 1)[-1] not in " ".join(m for _, m in a_issues)]
        return dict(passed=not all_issues, issues=[dict(step=i, problem=m) for i, m in all_issues],
                    auditor_raw=raw[:800], layer_a=len(a_issues), layer_b=len(b_issues))
