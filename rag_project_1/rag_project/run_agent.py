"""python run_agent.py            -> 10-query eval (outputs/agent_eval.csv, traces/) + seeded stress test (outputs/stress_eval.csv)
   python run_agent.py --ask "..." [--fault calculator:1:wrong]
Set LLM_PROVIDER/LLM_MODEL/LLM_API_KEY to run with a real model (default: scripted MockLLM)."""
import argparse, csv, json, os, re, sys
from pipeline import load_docs
from tools import Tools, ToolBox, FaultPlan
from llm import make_llm
from verifier import Verifier
from agent import Agent

ROOT = os.path.dirname(os.path.abspath(__file__))
docs = load_docs(os.path.join(ROOT, "corpus"))
TOOLS = Tools(docs)

# (id, category, question, gold regex (commas removed from answer first), forced faults, persistent faults)
Q = [
 (1, "simple", "What is the international travel per-diem?", r"9000", {}, {}),
 (2, "simple", "What is 18% GST on INR 12,500?", r"2250", {}, {}),
 (3, "multi-step", "What is the total per-diem allowance in INR for a 5-day international trip plus a 3-day domestic trip?", r"54600", {}, {}),
 (4, "multi-step", "What is the total per-diem allowance in USD for a 4-day international trip?", r"433\.7", {}, {}),
 (5, "multi-step", "Which is bigger: the home office equipment limit or 5 days of international per-diem, and by how much?", r"45000.*5000|5000.*45000", {}, {}),
 (6, "fault:wrong-calculator", "What is the total per-diem allowance in INR for a 6-day international trip plus a 2-day domestic trip?", r"60400", {("calculator", 1): "wrong"}, {}),
 (7, "fault:timeout+error", "What is the total per-diem allowance in USD for a 3-day international trip?", r"325\.3", {("lookup_policy", 1): "timeout", ("calculator", 1): "error", ("convert_currency", 1): "timeout"}, {}),
 (8, "fault:wrong-lookup", "What is the total per-diem allowance in INR for a 7-day international trip?", r"63000", {("lookup_policy", 1): "wrong"}, {}),
 (9, "fault:wrong-conversion", "What is the total per-diem allowance in USD for a 2-day international trip?", r"216\.8", {("convert_currency", 1): "wrong"}, {}),
 (10, "fault:tool-down", "What is the international travel per-diem?", r"could not|unable|unavailable", {}, {"lookup_policy": "timeout"}),
]


def correct(answer, gold):
    a = re.sub(r"(?<=\d),(?=\d)", "", answer or "").split("[UNVERIFIED")[0]
    if "INCOMPLETE" in a and "could not" not in gold: return False
    return bool(re.search(gold, a, re.I | re.S))


def run_one(llm, forced, persistent, question, verify=True, seed=0, rates=(0, 0, 0)):
    tb = ToolBox(TOOLS, FaultPlan(seed, *rates, forced=forced, persistent=persistent))
    return Agent(llm, tb, Verifier(llm, docs) if verify else None).run(question)


def render(t):
    L = [f"QUESTION: {t['question']}", f"FAULTS: {t['faults']}", ""]
    for s in t["steps"]:
        L.append(f"step {s['n'] if 'n' in s else '*'}  THOUGHT: {s['thought']}\n        ACTION: {s['action']} {json.dumps(s['input'])}\n        OBS: {s['obs'][:300]}" + ("   <-- RE-RUN" if s.get("rerun") else ""))
    L.append("")
    L += ["TOOL CALL LOG: " + "; ".join(f"{c['tool']}#{c['call_no']}[{c['injected']}]{'ok' if c['ok'] else 'FAIL'}" for c in t["tool_calls"]), ""]
    for e in t["events"]: L.append("EVENT: " + json.dumps(e))
    for i, v in enumerate(t["verifications"], 1):
        L.append(f"VERIFICATION {i}: {'PASS' if v['passed'] else 'FAIL'}  draft={v['draft']!r}\n   issues={json.dumps(v['issues'])}")
    L += ["", f"DRAFT ANSWER (before verification): {t['draft_answer']}", f"FINAL ANSWER: {t['final_answer']}", f"TERMINATED BY: {t['terminated_by']}  (llm steps={t['llm_steps']})"]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ask"); ap.add_argument("--fault", action="append", default=[])
    ap.add_argument("--no-verify", action="store_true"); args = ap.parse_args()
    llm = make_llm()
    if args.ask:
        forced = {(f.split(":")[0], int(f.split(":")[1])): f.split(":")[2] for f in args.fault}
        t = run_one(llm, forced, {}, args.ask, verify=not args.no_verify); print(render(t)); sys.exit()
    os.makedirs(os.path.join(ROOT, "traces"), exist_ok=True); os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    rows = []
    for qid, cat, q, gold, forced, pers in Q:
        t = run_one(llm, forced, pers, q, True)
        t0 = run_one(llm, forced, pers, q, False)          # same fault plan, verification OFF (ablation)
        c_final, c_draft, c_off = correct(t["final_answer"], gold), correct(t["draft_answer"], gold), correct(t0["final_answer"], gold)
        effect = "none" if not t["verification_changed_answer"] else ("helped" if c_final and not c_draft else "hurt" if c_draft and not c_final else "neutral")
        t.update(qid=qid, category=cat, gold=gold, correct=c_final, correct_without_verification=c_off, effect=effect, llm=llm.name)
        base = os.path.join(ROOT, "traces", f"q{qid:02d}_{cat.replace(':','-').replace('+','_')}")
        json.dump(t, open(base + ".json", "w"), indent=2); open(base + ".txt", "w").write(render(t) + "\n")
        rows.append(dict(qid=qid, category=cat, question=q, faults_injected=t["faults"], steps=t["llm_steps"],
                         tool_calls_incl_retries=len(t["tool_calls"]), tools_called=">".join(s["action"] for s in t["steps"]),
                         final_answer=t["final_answer"], correct=c_final, correct_without_verification=c_off, terminated_by=t["terminated_by"],
                         verification_fired=t["verification_fired"], verification_changed_answer=t["verification_changed_answer"], verification_effect=effect))
        print(f"Q{qid:<2} {cat:24} steps={t['llm_steps']} correct={c_final!s:5} (no-verify: {c_off!s:5}) fired={t['verification_fired']!s:5} effect={effect}")
    with open(os.path.join(ROOT, "outputs", "agent_eval.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("accuracy with verification   :", sum(r["correct"] for r in rows), "/", len(rows))
    print("accuracy without verification:", sum(r["correct_without_verification"] for r in rows), "/", len(rows))
    # ---- stress: random faults, 10 seeds x 10 questions, verification on vs off (paired fault draws)
    srows, tot, cnt = [], {True: 0, False: 0}, {}
    for seed in range(10):
        for qid, cat, q, gold, _, _ in Q[:9]:
            for ver in (True, False):
                t = run_one(llm, {}, {}, q, ver, seed * 100 + qid, (0.10, 0.10, 0.15))   # independent fault schedule per (seed, question)
                fa = t["final_answer"] or ""
                ok = correct(fa, gold); tot[ver] += ok
                flagged = bool(re.match(r"\s*(I could not|INCOMPLETE)", fa)) or "[UNVERIFIED" in fa
                outcome = "correct" if ok else "honest_failure" if flagged else "SILENT_WRONG"
                cnt[(ver, outcome)] = cnt.get((ver, outcome), 0) + 1
                srows.append(dict(seed=seed, qid=qid, verify=ver, correct=ok, outcome=outcome, terminated_by=t["terminated_by"], fired=t["verification_fired"], faults_hit=";".join(f"{c['tool']}#{c['call_no']}:{c['injected']}" for c in t["tool_calls"] if c["injected"]!="ok"), final_answer=(t["final_answer"] or "")[:110]))
    with open(os.path.join(ROOT, "outputs", "stress_eval.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0])); w.writeheader(); w.writerows(srows)
    n = 90
    print(f"STRESS (random faults p_timeout=.10 p_error=.10 p_wrong=.15; {n} runs each): verify ON {tot[True]}/{n}  verify OFF {tot[False]}/{n}")
    for ver in (True, False):
        print(f"   verify={'ON ' if ver else 'OFF'}: " + ", ".join(f"{o}={cnt.get((ver, o), 0)}" for o in ("correct", "honest_failure", "SILENT_WRONG")))
    open(os.path.join(ROOT, "outputs", "agent_run_info.txt"), "w").write(f"LLM: {llm.name}\nStress: ON {tot[True]}/{n}, OFF {tot[False]}/{n}\n" +
        "\n".join(f"verify={v}: " + ", ".join(f"{o}={cnt.get((v, o), 0)}" for o in ("correct", "honest_failure", "SILENT_WRONG")) for v in (True, False)) + "\n")
