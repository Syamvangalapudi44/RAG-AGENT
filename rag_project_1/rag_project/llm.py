"""LLM access for the agent and the verifier (same env vars as Q1: LLM_PROVIDER / LLM_MODEL / LLM_API_KEY / LLM_BASE_URL).
Without LLM_PROVIDER a deterministic *scripted* MockLLM is used so the whole project runs offline. The mock is NOT
an LLM: its planner is hand-written for the 10 evaluation questions and its 'auditor' only echoes the programmatic
findings. Real behaviour (tool selection, free-form verification) can only be measured with a real model.
"""
import json, os, re, urllib.request


class LLMClient:
    name = "llm"

    def chat(self, system, messages):
        p = os.environ["LLM_PROVIDER"].lower()
        model = os.environ.get("LLM_MODEL", {"openai": "gpt-4o-mini", "ollama": "llama3", "anthropic": "claude-haiku-4-5"}[p])
        key = os.environ.get("LLM_API_KEY", "")
        self.name = f"{p}/{model}"
        if p == "anthropic":
            req = urllib.request.Request(os.environ.get("LLM_BASE_URL", "https://api.anthropic.com") + "/v1/messages", method="POST",
                data=json.dumps({"model": model, "max_tokens": 600, "temperature": 0, "system": system, "messages": messages}).encode(),
                headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
            return json.load(urllib.request.urlopen(req, timeout=60))["content"][0]["text"].strip()
        base = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1" if p == "ollama" else "https://api.openai.com/v1")
        req = urllib.request.Request(base + "/chat/completions", method="POST",
            data=json.dumps({"model": model, "temperature": 0, "messages": [{"role": "system", "content": system}] + messages}).encode(),
            headers={"content-type": "application/json", "authorization": "Bearer " + key})
        return json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"].strip()


def make_llm():
    return LLMClient() if os.environ.get("LLM_PROVIDER", "").lower() in ("openai", "ollama", "anthropic") else MockLLM()


# ------------------------------------------------------------------------------------------------ scripted stand-in
def _fmt(v):
    return f"{v:,}" if isinstance(v, int) else str(v)


class MockLLM:
    name = "MOCK (scripted planner, not an LLM)"

    def chat(self, system, messages):
        if system.startswith("You are an independent auditor"):
            body = messages[-1]["content"]
            tail = body.split("PROGRAMMATIC FINDINGS:")[-1].strip()
            if tail.lower().startswith("none"):
                return json.dumps({"verdict": "pass", "issues": []})
            issues = [{"step": int(m.group(1)), "problem": m.group(2)} for m in re.finditer(r"- step (\d+): (.*)", tail)]
            return json.dumps({"verdict": "fail", "issues": issues or [{"step": None, "problem": tail[:200]}]})
        q = messages[0]["content"].split("Question:", 1)[1].strip()
        obs = [(m.group(1), m.group(2)) for msg in messages if msg["role"] == "user"
               for m in [re.match(r"Observation \[tool=(\w+)\]: (.*)", msg["content"], re.S)] if m]
        step = lambda th, a, i: json.dumps({"thought": th, "action": a, "action_input": i})
        if obs and obs[-1][1].startswith("ERROR"):
            return step("The tool failed repeatedly; I should not guess.", "finish",
                        {"answer": f"I could not complete this request: the {obs[-1][0]} tool was unavailable ({obs[-1][1][:60]}).", "sources": []})
        ql, n = q.lower(), len(obs)
        val = lambda i: json.loads(obs[i][1])["result"]

        def perdiem(i, kind):
            for r in json.loads(obs[i][1])["results"]:
                m = re.search(kind + r" travel per-diem is INR ([\d,]+)", r["text"], re.I)
                if m: return int(m.group(1).replace(",", "")), r["file"]
            return None, None

        m = re.search(r"(\d+(?:\.\d+)?)% gst on inr ([\d,]+)", ql)
        if m:
            if n == 0: return step("Percentage arithmetic -> calculator.", "calculator", {"expression": f"{m.group(2)}*{m.group(1)}/100"})
            return step("Done.", "finish", {"answer": f"GST at {m.group(1)}% on INR {m.group(2)} is INR {_fmt(val(0))}.", "sources": []})
        if "bigger" in ql:
            d = int(re.search(r"(\d+) days? of international", ql).group(1))
            if n == 0: return step("Need the equipment limit from policy.", "lookup_policy", {"query": "home office equipment limit"})
            if n == 1: return step("Need the per-diem from policy.", "lookup_policy", {"query": "international travel per-diem"})
            lim_m = re.search(r"INR ([\d,]+)", json.loads(obs[0][1])["results"][0]["text"])
            pd, f2 = perdiem(1, "international")
            if lim_m is None or pd is None: return step("Lookup lacked the numbers.", "finish", {"answer": "I could not find the values needed.", "sources": []})
            lim = int(lim_m.group(1).replace(",", ""))
            if n == 2: return step("Total for the days.", "calculator", {"expression": f"{d}*{pd}"})
            if n == 3: return step("Compare with the limit.", "calculator", {"expression": f"{val(2)}-{lim}"})
            f1 = json.loads(obs[0][1])["results"][0]["file"]
            diff = val(3)
            return step("Done.", "finish", {"answer": f"{d} days of international per-diem (INR {_fmt(val(2))}) is " + ("bigger" if diff > 0 else "smaller") +
                                              f" than the home office equipment limit (INR {_fmt(lim)}), by INR {_fmt(abs(diff))}.", "sources": [f1, f2]})
        legs = re.findall(r"(\d+)-day (international|domestic)", ql)
        if legs:
            kinds = list(dict.fromkeys(k for _, k in legs))
            if n < len(kinds): return step(f"Need the {kinds[n]} per-diem from policy.", "lookup_policy", {"query": f"{kinds[n]} travel per-diem"})
            rates, files = {}, []
            for i, k in enumerate(kinds):
                rates[k], f = perdiem(i, k); files.append(f)
            if None in rates.values(): return step("Lookup lacked the rate.", "finish", {"answer": "I could not find the per-diem rate.", "sources": []})
            if n == len(kinds): return step("Compute total.", "calculator", {"expression": " + ".join(f"{d}*{rates[k]}" for d, k in legs)})
            usd = "usd" in ql
            if usd and n == len(kinds) + 1: return step("Convert to USD.", "convert_currency", {"amount": val(len(kinds)), "from": "INR", "to": "USD"})
            tot = val(len(kinds))
            ans = f"The total per-diem allowance is USD {_fmt(val(len(kinds)+1))} (INR {_fmt(tot)} at {83} INR per USD)." if usd else f"The total per-diem allowance is INR {_fmt(tot)}."
            return step("Done.", "finish", {"answer": ans, "sources": files})
        if n == 0: return step("Policy fact -> lookup.", "lookup_policy", {"query": q})
        r = json.loads(obs[0][1])["results"][0]
        return step("Done.", "finish", {"answer": r["text"], "sources": [r["file"]]})
