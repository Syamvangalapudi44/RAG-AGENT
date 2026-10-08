"""Tools + deliberate unreliability.

Tools:  calculator        - safe arithmetic evaluator
        lookup_policy     - the Q1 retriever (structure chunks + conflict resolution), retrieval only
        convert_currency  - INR<->USD at a fixed demo rate (83.0 INR per USD)
FaultPlan wraps every call and, under a seed, can: TIMEOUT (no response), ERROR (HTTP-style error or
truncated/malformed JSON) or WRONG (a plausible, well-formed, but incorrect value - nothing signals failure).
Faults are drawn per (seed, tool, n-th call of that tool), so the same plan hits the same calls whether or not
verification is on (paired comparison).
"""
import ast, json, operator as op, random, re, threading, time
from pipeline import RAG, load_docs

INR_PER_USD = 83.0
TIMEOUT_S = 0.15
HANG_S = 1.0

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod}


def safe_eval(expr):
    expr = expr.replace(",", "").replace("^", "**")
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        raise ValueError("unsupported expression")
    v = ev(ast.parse(expr.strip(), mode="eval"))
    return int(v) if float(v).is_integer() else round(v, 6)


class Tools:
    def __init__(self, docs=None):
        self.docs = docs or load_docs()
        self.rag = RAG(self.docs, "structure", "resolved")

    def calculator(self, a):
        return {"result": safe_eval(str(a["expression"]))}

    def convert_currency(self, a):
        amt, f, t = float(str(a["amount"]).replace(",", "")), a["from"].upper(), a["to"].upper()
        if (f, t) == ("INR", "USD"): r = amt / INR_PER_USD
        elif (f, t) == ("USD", "INR"): r = amt * INR_PER_USD
        else: raise ValueError("only INR<->USD supported")
        return {"result": round(r, 2), "rate_inr_per_usd": INR_PER_USD}

    def lookup_policy(self, a):
        final, notes = self.rag.retrieve(str(a["query"]), k=2)
        out = []
        for c, s in final:
            raw = self.docs[c.file][c.start:c.end]
            out.append({"file": c.file, "header": c.header, "text": re.sub(r"\s+", " ", c.body).strip(),
                        "source_excerpt": re.sub(r"\s+", " ", raw).strip(), "score": round(s, 3)})
        return {"results": out, "notes": notes}


class FaultPlan:
    def __init__(self, seed=0, p_timeout=0.0, p_error=0.0, p_wrong=0.0, forced=None, persistent=None):
        self.seed, self.p = seed, (p_timeout, p_error, p_wrong)
        self.forced, self.persistent, self.count = dict(forced or {}), dict(persistent or {}), {}

    def decide(self, tool):
        n = self.count[tool] = self.count.get(tool, 0) + 1
        if tool in self.persistent: return self.persistent[tool], n
        if (tool, n) in self.forced: return self.forced[(tool, n)], n
        u = random.Random(f"{self.seed}-{tool}-{n}").random()
        pt, pe, pw = self.p
        return ("timeout" if u < pt else "error" if u < pt + pe else "wrong" if u < pt + pe + pw else "ok"), n

    def describe(self):
        d = [f"{k[0]}#{k[1]}={v}" for k, v in self.forced.items()] + [f"{k}=ALWAYS {v}" for k, v in self.persistent.items()]
        if any(self.p): d.append(f"random(seed={self.seed}, timeout={self.p[0]}, error={self.p[1]}, wrong={self.p[2]})")
        return "; ".join(d) or "none"


def corrupt(tool, payload, rng):
    """plausible-but-wrong: well-formed, same shape, subtly different value"""
    if tool == "calculator" or tool == "convert_currency":
        payload = dict(payload)
        f = rng.choice([1.04, 0.96, 1.10, 0.90, 1.02])
        v = payload["result"] * f
        payload["result"] = round(v) if isinstance(payload["result"], int) else round(v, 2)
        return payload
    if tool == "lookup_policy":
        payload = json.loads(json.dumps(payload))
        for r in payload["results"][:1]:           # tamper with the best chunk's number
            def bump(m):
                digits = list(m.group(0))
                i = next(i for i in range(len(digits)) if digits[i].isdigit() and digits[i] != "0")
                digits[i] = str((int(digits[i]) + rng.choice([1, 2, 3])) % 10 or 1)
                return "".join(digits)
            r["text"] = re.sub(r"(?<=INR )[\d,]+", bump, r["text"], count=1)
        return payload
    return payload


def _validate(tool, payload):
    if tool in ("calculator", "convert_currency"):
        return isinstance(payload, dict) and isinstance(payload.get("result"), (int, float))
    if tool == "lookup_policy":
        return isinstance(payload, dict) and isinstance(payload.get("results"), list)
    return False


class ToolBox:
    def __init__(self, tools, plan):
        self.tools, self.plan, self.log = tools, plan, []

    def call(self, name, args):
        """returns dict(ok, obs, injected). Never raises; never blocks longer than TIMEOUT_S."""
        kind, n = self.plan.decide(name)
        rec = dict(tool=name, call_no=n, injected=kind, args=args)
        box = {}

        def work():
            try:
                if kind == "timeout":
                    time.sleep(HANG_S); return
                payload = getattr(self.tools, name)(args)
                if kind == "wrong":
                    payload = corrupt(name, payload, random.Random(f"w-{self.plan.seed}-{name}-{n}"))
                box["raw"] = json.dumps(payload)
                if kind == "error":
                    box["raw"] = random.Random(f"e-{self.plan.seed}-{name}-{n}").choice(
                        ["ERROR: 503 Service Unavailable", box["raw"][: max(6, len(box["raw"]) // 2)]])
            except Exception as e:
                box["raw"] = f"ERROR: {type(e).__name__}: {e}"

        th = threading.Thread(target=work, daemon=True); th.start(); th.join(TIMEOUT_S)
        if "raw" not in box:
            rec.update(ok=False, obs=f"ERROR: TIMEOUT - no response from {name} within {TIMEOUT_S}s")
        else:
            raw = box["raw"]
            try:
                ok = (not raw.startswith("ERROR")) and _validate(name, json.loads(raw))
            except json.JSONDecodeError:
                ok, raw = False, "ERROR: malformed response (invalid JSON): " + raw[:40]
            rec.update(ok=ok, obs=raw)
        self.log.append(rec)
        return rec
