"""ReAct agent: structured {thought, action, action_input} steps, tool use chosen by the model, self-termination via `finish`.

Termination strategy
  * normal: the model emits action `finish` (and the verifier passes, or the verification budget is spent -> answer flagged UNVERIFIED)
  * hard caps (every one is recorded in trace['terminated_by']):  step_cap (max LLM steps), loop_detected (same action+input 3x),
    malformed_output (3 unparseable replies).  On any cap the agent returns an explicit 'INCOMPLETE ...' answer instead of guessing.
Tool failures: a timeout/error is retried (max_retries) by the harness; if it still fails the model sees 'ERROR: ...' and must
  say it cannot answer rather than invent a value.
Verification: on `finish`, a separate verifier audits the draft. If it fails, the earliest flagged step is RE-RUN (history after it is
  discarded so nothing downstream keeps the bad value), the event is recorded in trace['events'], and the model continues.
"""
import json, re
from tools import ToolBox

SYSTEM = """You are a careful assistant that answers questions about company policy and does arithmetic, using tools.
Tools (action names):
  calculator        action_input {"expression": "5*9000 + 3*3200"}   - ALL arithmetic goes through this tool
  lookup_policy     action_input {"query": "international travel per-diem"} - searches the company policy documents; never answer policy facts from memory
  convert_currency  action_input {"amount": 36000, "from": "INR", "to": "USD"}
  finish            action_input {"answer": "...", "sources": ["file.md", ...]} - give the final answer
Every reply must be ONE JSON object and nothing else: {"thought": "...", "action": "<tool or finish>", "action_input": {...}}
Rules: call one tool per step; choose tools by what the question needs; use only values returned by tools; if a tool reports ERROR or
is unavailable, do not guess - finish with an answer saying you could not complete it; cite the source files for policy facts."""

TOOL_ARGS = {"calculator": ["expression"], "lookup_policy": ["query"], "convert_currency": ["amount", "from", "to"], "finish": ["answer"]}


def parse_action(raw):
    try:
        d = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        assert {"thought", "action", "action_input"} <= set(d)
    except Exception:
        return None
    a, i = d["action"], d["action_input"]
    if isinstance(i, str) and a in TOOL_ARGS:
        i = {TOOL_ARGS[a][0]: i}
    if not isinstance(i, dict):
        return None
    d["action_input"] = i
    return d


class Agent:
    def __init__(self, llm, toolbox, verifier=None, max_steps=10, max_retries=2, max_verify_rounds=2):
        self.llm, self.tb, self.verifier = llm, toolbox, verifier
        self.max_steps, self.max_retries, self.max_verify_rounds = max_steps, max_retries, max_verify_rounds

    def _exec(self, action, inp, events):
        for attempt in range(1 + self.max_retries):
            r = self.tb.call(action, inp)
            if r["ok"]:
                return True, r["obs"]
            events.append(dict(type="retry", tool=action, attempt=attempt + 1, injected=r["injected"], reason=r["obs"][:80]))
        return False, f"ERROR: {action} unavailable after {1 + self.max_retries} attempts (last: {r['obs'][:70]})"

    def run(self, question):
        messages = [{"role": "user", "content": f"Question: {question}"}]
        steps, events, verifs, seen = [], [], [], {}
        draft_first, final, term, n_llm, vrounds, bad = None, None, None, 0, 0, 0
        while True:
            if n_llm >= self.max_steps:
                term, final = "step_cap", "INCOMPLETE: step limit reached before a verified answer. Partial tool results: " + \
                    "; ".join(f"{s['action']}->{s['obs'][:40]}" for s in steps)
                break
            raw = self.llm.chat(SYSTEM, messages); n_llm += 1
            act = parse_action(raw)
            if act is None or (act["action"] not in TOOL_ARGS) or not all(k in act["action_input"] for k in TOOL_ARGS[act["action"]]):
                bad += 1
                events.append(dict(type="malformed", raw=raw[:100]))
                if bad > 2:
                    term, final = "malformed_output", "INCOMPLETE: the model did not produce valid structured output."; break
                messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": "Observation [tool=none]: ERROR: reply must be one valid JSON object with a known action and its required action_input keys."}]
                continue
            if act["action"] == "finish":
                ans = act["action_input"]["answer"]; srcs = act["action_input"].get("sources", [])
                if draft_first is None: draft_first = ans
                if self.verifier is None:
                    term, final = "finish", ans; break
                v = self.verifier.verify(question, steps, ans, srcs)
                v["draft"] = ans; verifs.append(v)
                if v["passed"]:
                    term, final = "finish", ans; break
                if vrounds >= self.max_verify_rounds:
                    term, final = "finish_unverified", ans + " [UNVERIFIED: " + "; ".join(i["problem"][:80] for i in v["issues"]) + "]"; break
                vrounds += 1
                flagged = sorted({i["step"] - 1 for i in v["issues"] if i["step"]})
                if flagged:
                    e = flagged[0]; st = steps[e]
                    events.append(dict(type="verification_rerun", step=e + 1, tool=st["action"], reason=[i["problem"] for i in v["issues"] if i["step"] == e + 1],
                                       old_obs=st["obs"][:200], discarded_later_steps=len(steps) - e - 1))
                    messages = messages[: st["msg_index"]]; steps = steps[:e]
                    ok, obs = self._exec(st["action"], st["input"], events)
                    steps.append(dict(st, ok=ok, obs=obs, rerun=True, msg_index=len(messages)))
                    messages += [{"role": "assistant", "content": json.dumps({"thought": st["thought"], "action": st["action"], "action_input": st["input"]})},
                                 {"role": "user", "content": f"Observation [tool={st['action']}]: {obs}"}]
                else:
                    events.append(dict(type="verification_rerun", step=None, reason=[i["problem"] for i in v["issues"]]))
                    messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": "VERIFIER FEEDBACK: " + "; ".join(i["problem"] for i in v["issues"]) +
                                 " Revise your answer using only tool outputs."}]
                continue
            key = (act["action"], json.dumps(act["action_input"], sort_keys=True))
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 4:
                term, final = "loop_detected", "INCOMPLETE: repeated the same action without progress."; break
            ok, obs = self._exec(act["action"], act["action_input"], events)
            steps.append(dict(n=len(steps) + 1, thought=act["thought"], action=act["action"], input=act["action_input"], ok=ok, obs=obs, msg_index=len(messages)))
            messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": f"Observation [tool={act['action']}]: {obs}"}]
        fa = final
        return dict(question=question, final_answer=fa, draft_answer=draft_first, terminated_by=term, llm_steps=n_llm, steps=steps,
                    tool_calls=list(self.tb.log), events=events, verifications=verifs, faults=self.tb.plan.describe(),
                    verification_fired=any(not v["passed"] for v in verifs),
                    verification_changed_answer=bool(draft_first is not None and fa is not None and re.sub(r"\W+", "", fa.lower()) != re.sub(r"\W+", "", draft_first.lower())))
