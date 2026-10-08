import os
from datetime import date
from chunkers import CHUNKERS
from embedder import Index, make_embedder, tokenize
from meta import extract
from resolver import parse_as_of, resolve
from generator import generate, ABSTAIN_TEXT

TODAY = date.fromisoformat(os.environ.get("AS_OF_TODAY", date.today().isoformat()))


def load_docs(path="corpus"):
    docs = {}
    for f in sorted(os.listdir(path)):
        with open(os.path.join(path, f), encoding="utf-8", newline="") as fh:
            docs[f] = fh.read()
    return docs


class RAG:
    def __init__(self, docs, strategy, mode, top_k=3, n_candidates=8, threshold=None):
        """mode: 'baseline' (naive top-k) | 'resolved' (metadata resolution + abstention)"""
        self.docs, self.strategy, self.mode, self.k, self.nc = docs, strategy, mode, top_k, n_candidates
        self.chunks = CHUNKERS[strategy](docs)
        self.embedder = make_embedder()
        self.index = Index(self.chunks, self.embedder)
        self.metas = extract(docs)
        self.by_file = {}
        for c in self.chunks:
            self.by_file.setdefault(c.file, []).append(c)
        self.threshold = self.embedder.default_threshold if threshold is None else threshold

    min_cover = 0.30   # abstain when < 30% of the query's idf-weighted terms appear in the evidence

    def coverage(self, query, final):
        """idf-weighted share of query terms present in the retrieved evidence. Query terms that
        are out-of-vocabulary for the whole corpus count at maximum weight (they signal the corpus
        simply does not talk about this topic)."""
        q = set(tokenize(query))
        ev = set(tokenize(" ".join(c.text for c, _ in final)))
        tot = sum(self.embedder.idf_of(t) for t in q) or 1.0
        return sum(self.embedder.idf_of(t) for t in q if t in ev) / tot

    def ask(self, query):
        notes, rawnotes, conflict = [], [], {}
        if self.mode == "baseline":
            final = self.index.search(query, self.k)
        else:
            as_of, explicit = parse_as_of(query, TODAY)
            cands = self.index.search(query, self.nc)
            final, rawnotes, conflict = resolve(cands, self.metas, self.docs, self.by_file, as_of, explicit, self.k)
            used = {c.file for c, _ in final}
            notes = [t for t, rel in rawnotes if rel & used]   # only provenance notes about evidence in play
        top = final[0][1] if final else 0.0
        if self.mode == "resolved" and conflict:
            txt = "Sources conflict and I cannot tell which governs: " + "; ".join(f"{t}: {', '.join(f)}" for t, f in conflict.items())
            return dict(answer=txt, sources=[f for fs in conflict.values() for f in fs], abstained=True, notes=notes, retrieved=final)
        cover = self.coverage(query, final)
        if self.mode == "resolved" and (top < self.threshold or cover < self.min_cover):
            return dict(answer=ABSTAIN_TEXT, sources=[], abstained=True, notes=notes, retrieved=final)
        ans, files, abst = generate(query, final, notes, self.embedder, min_cover=0.0)
        if self.mode == 'resolved':
            notes = [t for t, rel in rawnotes if rel & set(files)]
        if notes and not abst and "NOTE:" not in ans:
            ans = ans + "\nNOTE: " + " ".join(notes)
        return dict(answer=ans, sources=files, abstained=abst, notes=notes, retrieved=final, coverage=cover)


def _retrieve(self, query, k=2):
    """Retrieval only (no generation): resolved retrieval used by the agent's lookup tool."""
    as_of, explicit = parse_as_of(query, TODAY)
    cands = self.index.search(query, self.nc)
    final, rawnotes, conflict = resolve(cands, self.metas, self.docs, self.by_file, as_of, explicit, k)
    return final, [t for t, rel in rawnotes if rel & {c.file for c, _ in final}]

RAG.retrieve = _retrieve
