"""Evaluation harness: 8 fixed queries, gold checks, error taxonomy, metrics, tables, plot."""
import csv, json, re, os
import numpy as np

QUERIES = [
 dict(id=1, q="What is the international travel per-diem?",
      must=[r"9,?000"], forbid=[r"6,?500"], files={"travel_policy_2024.md"}, evidence=r"9,000", ev_files={"travel_policy_2024.md"},
      gold="INR 9,000/day (travel_policy_2024.md; 2019 value INR 6,500 is superseded)"),
 dict(id=2, q="What is the current limit for home office equipment?",
      must=[r"40,?000"], forbid=[r"25,?000"], files={"memo_2024_08.md"}, evidence=r"40,000", ev_files={"memo_2024_08.md"},
      gold="INR 40,000 per rolling 12 months (memo_2024_08.md amends clause 5.2)"),
 dict(id=3, q="I am a contractor. Will the company buy me a laptop?",
      must=[r"not covered|supply their own|must supply"], forbid=[], neg=r"provides a laptop|will buy|\byes\b",
      files={"equipment_policy.md"}, evidence=r"Contractors", ev_files={"equipment_policy.md"},
      gold="No - contractors are not covered and must supply their own equipment (clause 5.3)"),
 dict(id=4, q="What notice period applies to a Lead?",
      must=[r"90 days"], forbid=[r"30 days|60 days"], files={"notice_periods.md"}, evidence=r"Lead", ev_files={"notice_periods.md"},
      gold="90 days (Band 4, Lead / Manager)"),
 dict(id=5, q="Above what amount do I need to keep a receipt?",
      must=[r"2,?500"], forbid=[r"\b1,?000\b"], files={"travel_policy_2024.md"}, evidence=r"2,500", ev_files={"travel_policy_2024.md"},
      gold="INR 2,500 (2024 policy; 2019 value INR 1,000 is superseded)"),
 dict(id=6, q="What is the per-diem for a trip taken in June 2019?",
      must=[r"2,?000", r"6,?500"], forbid=[r"3,?200|9,?000"], files={"travel_policy_2019.md"}, evidence=r"6,500|2,000",
      ev_files={"travel_policy_2019.md"}, past=True,
      gold="Domestic INR 2,000 and international INR 6,500 per day (2019 policy was in force)"),
 dict(id=7, q="Who approves business class travel?",
      must=[r"Director"], forbid=[], files={"travel_policy_2019.md", "travel_policy_2024.md"}, evidence=r"Business class",
      ev_files={"travel_policy_2019.md", "travel_policy_2024.md"},
      gold="A Director, in writing (identical in both policy versions)"),
 dict(id=8, q="What is the company's parental leave entitlement?",
      must=[], forbid=[], files=set(), abstain=True, evidence=None, ev_files=set(),
      gold="Not in the corpus -> the system should abstain"),
]

ABSTAIN_RE = re.compile(r"i don't know|do not contain|sources conflict", re.I)


def main_answer(a):
    return a.split("\nNOTE:")[0]


def is_relevant(ch, spec):
    return spec["evidence"] is not None and ch.file in spec["ev_files"] and re.search(spec["evidence"], ch.body)


def judge(spec, out, corpus_texts):
    """returns (correct: bool, error_type or None)"""
    ans = main_answer(out["answer"])
    abst = out["abstained"]
    if spec.get("abstain"):
        return (abst, None if abst else "fabricated")
    if abst:
        return False, "false_abstention"
    ok_content = all(re.search(m, ans, re.I) for m in spec["must"]) and not any(re.search(f, ans) for f in spec["forbid"])
    ok_cite = bool(set(out["sources"]) & spec["files"])
    if ok_content and ok_cite:
        return True, None
    # ---- classify the failure (rule-based; every label is re-read manually in ERROR_ANALYSIS.md)
    ret = out["retrieved"]
    if spec.get("neg") and re.search(spec["neg"], ans, re.I) and not re.search(spec["must"][0], ans, re.I):
        return False, "negation"
    if any(re.search(f, ans) for f in spec["forbid"]):
        bearing = [c for c, _ in ret if is_relevant(c, spec)]
        if spec.get("past") or bearing:        # the right chunk was retrieved too, the wrong one won / both given
            return False, "conflict"
        return False, "stale"
    in_ev_file = [c for c, _ in ret if c.file in spec["ev_files"]]
    if in_ev_file and not any(is_relevant(c, spec) for c, _ in ret):
        return False, "fragment"
    if in_ev_file and not all(re.search(m, ans, re.I) for m in spec["must"]):
        return False, "fragment"
    return False, "fabricated"


def run_eval(make_rag, docs, strategies=("fixed", "structure"), modes=("baseline", "resolved")):
    rows, summary = [], []
    for st in strategies:
        for mode in modes:
            rag = make_rag(st, mode)
            sizes = [len(c.text) for c in rag.chunks]
            correct = abst = 0
            p3, p3n = 0.0, 0
            for spec in QUERIES:
                out = rag.ask(spec["q"])
                ok, err = judge(spec, out, docs)
                correct += ok
                abst += out["abstained"]
                if spec["evidence"]:
                    top = out["retrieved"][:3]
                    p3 += sum(bool(is_relevant(c, spec)) for c, _ in top) / 3
                    p3n += 1
                rows.append(dict(strategy=st, mode=mode, qid=spec["id"], query=spec["q"], gold=spec["gold"],
                                 answer=out["answer"].replace("\n", " | "), cited=";".join(out["sources"]),
                                 retrieved=";".join(f"{c.cid}({s:.2f})" for c, s in out["retrieved"]),
                                 correct=ok, error_type=err or ""))
            summary.append(dict(strategy=st, mode=mode, accuracy=correct / len(QUERIES), correct=correct,
                                p_at_3=p3 / p3n, n_chunks=len(sizes), mean_chars=float(np.mean(sizes)),
                                median_chars=float(np.median(sizes)), min_chars=min(sizes), max_chars=max(sizes),
                                abstention_rate=abst / len(QUERIES), sizes=sizes))
    return rows, summary


def write_outputs(rows, summary, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(f"{outdir}/results.json", "w", encoding="utf-8") as f:
        json.dump(dict(summary=[{k: v for k, v in s.items() if k != "sizes"} for s in summary], rows=rows), f, indent=2)
    md = ["| Chunking | Pipeline | Accuracy | P@3 | Chunks | Mean size (chars) | Abstention rate |", "|---|---|---|---|---|---|---|"]
    for s in summary:
        md.append(f"| {s['strategy']} | {s['mode']} | {s['correct']}/8 = {s['accuracy']:.0%} | {s['p_at_3']:.2f} | {s['n_chunks']} | "
                  f"{s['mean_chars']:.0f} | {s['abstention_rate']:.0%} |")
    md += ["", "### Per-query outcome (✓ correct / error type)", "",
           "| Q | " + " | ".join(f"{s['strategy']}/{s['mode']}" for s in summary) + " |", "|---|" + "---|" * len(summary)]
    for spec in QUERIES:
        cells = []
        for s in summary:
            r = next(r for r in rows if r["strategy"] == s["strategy"] and r["mode"] == s["mode"] and r["qid"] == spec["id"])
            cells.append("✓" if r["correct"] else r["error_type"])
        md.append(f"| {spec['id']} | " + " | ".join(cells) + " |")
    open(f"{outdir}/comparison.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    # chunk size plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    seen = {}
    for s in summary:
        seen.setdefault(s["strategy"], s["sizes"])
    bins = np.arange(0, max(max(v) for v in seen.values()) + 40, 25)
    for a, (st, sizes) in zip(ax, seen.items()):
        a.hist(sizes, bins=bins, color="#4C72B0" if st == "fixed" else "#DD8452", edgecolor="white")
        a.axvline(np.mean(sizes), color="k", ls="--", label=f"mean {np.mean(sizes):.0f}")
        a.set_title(f"{st} chunking  (n={len(sizes)})"); a.set_xlabel("chunk size (characters, incl. header)"); a.set_ylabel("# chunks"); a.legend()
    plt.tight_layout(); plt.savefig(f"{outdir}/chunk_size_distribution.png", dpi=140); plt.close()
    return "\n".join(md)
