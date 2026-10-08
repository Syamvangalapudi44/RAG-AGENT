"""Run end-to-end:  python run.py            -> full evaluation (4 configurations x 8 queries)
                   python run.py --ask "..." -> ask one question (structure chunking + conflict resolution)
LLM: export LLM_PROVIDER=openai|ollama|anthropic  LLM_MODEL=...  LLM_API_KEY=...  [LLM_BASE_URL=...]
     (unset -> offline extractive generator; results are labelled accordingly)
"""
import argparse, os, sys
from pipeline import RAG, load_docs
from generator import llm_available
from evaluate import run_eval, write_outputs

ap = argparse.ArgumentParser()
ap.add_argument("--ask")
ap.add_argument("--strategy", default="structure", choices=["fixed", "structure"])
ap.add_argument("--mode", default="resolved", choices=["baseline", "resolved"])
args = ap.parse_args()
docs = load_docs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))

if args.ask:
    o = RAG(docs, args.strategy, args.mode).ask(args.ask)
    print(o["answer"], "\nSources:", ", ".join(o["sources"]) or "-")
    sys.exit()

gen = ("LLM: %s / %s" % (os.environ["LLM_PROVIDER"], os.environ.get("LLM_MODEL", "default"))) if llm_available() \
      else "OFFLINE extractive generator (no LLM configured)"
print("Generator:", gen)
rows, summary = run_eval(lambda st, mode: RAG(docs, st, mode), docs)
print(write_outputs(rows, summary, os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "generator_used.txt"), "w").write(gen + "\n")
print("\nWrote results/comparison.md, per_query.csv, results.json, chunk_size_distribution.png")
