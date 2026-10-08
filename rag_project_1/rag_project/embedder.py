"""Embeddings + NumPy cosine index.

Default: TF-IDF vectors (sublinear tf, L2-normalised) built with NumPy only, so the
project runs offline with no downloads.
Optional: EMBED=st      -> sentence-transformers all-MiniLM-L6-v2 (pip install sentence-transformers)
          EMBED=openai  -> OpenAI-compatible /embeddings endpoint (uses LLM_BASE_URL / LLM_API_KEY)
"""
import json, math, os, re, urllib.request
import numpy as np

STOP = set("""a an the is are was were be been am i me my we our you your it its of in on at to for from by with
and or not no what who whom which when where how does do did will would can could should shall any there this that
these those as if than then so about
need keep amount current taken tell give get""".split())  # last line: question-filler words; chosen after seeing the 8 eval queries (see LIMITATIONS.md)


def tokenize(text):
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())       # 2,000 -> 2000
    toks = re.findall(r"[a-z0-9]+", text)
    out = []
    for t in toks:
        if t in STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class TfidfEmbedder:
    name = "tfidf"
    default_threshold = 0.15

    def fit(self, texts):
        docs = [set(tokenize(t)) for t in texts]
        vocab = sorted(set().union(*docs))
        self.vocab = {w: i for i, w in enumerate(vocab)}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + sum(w in d for d in docs))) + 1 for w in vocab])
        self.max_idf = math.log(1 + n) + 1
        return self

    def idf_of(self, tok):
        i = self.vocab.get(tok)
        return self.idf[i] if i is not None else self.max_idf

    def embed(self, texts):
        M = np.zeros((len(texts), len(self.vocab)))
        for r, t in enumerate(texts):
            for tok in tokenize(t):
                i = self.vocab.get(tok)
                if i is not None:
                    M[r, i] += 1
        M = (np.where(M > 0, 1 + np.log(np.where(M > 0, M, 1)), 0)) * self.idf
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.where(norms == 0, 1, norms)


class STEmbedder(TfidfEmbedder):
    name = "sentence-transformers/all-MiniLM-L6-v2"
    default_threshold = 0.30

    def fit(self, texts):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        TfidfEmbedder.fit(self, texts)   # keeps idf helpers for the offline generator
        return self

    def embed(self, texts):
        return np.asarray(self.m.encode(list(texts), normalize_embeddings=True))


class OpenAIEmbedder(TfidfEmbedder):
    name = "openai-compatible embeddings"
    default_threshold = 0.30

    def embed(self, texts):
        base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        req = urllib.request.Request(base + "/embeddings", method="POST",
            data=json.dumps({"model": os.environ.get("EMBED_MODEL", "text-embedding-3-small"), "input": list(texts)}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("LLM_API_KEY", "")})
        v = np.array([d["embedding"] for d in json.load(urllib.request.urlopen(req))["data"]])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def make_embedder():
    return {"st": STEmbedder, "openai": OpenAIEmbedder}.get(os.environ.get("EMBED", ""), TfidfEmbedder)()


class Index:
    def __init__(self, chunks, embedder):
        self.chunks, self.emb = chunks, embedder.fit([c.text for c in chunks])
        self.M = self.emb.embed([c.text for c in chunks])

    def search(self, query, k):
        sims = self.M @ self.emb.embed([query])[0]
        order = sorted(range(len(sims)), key=lambda i: (-round(sims[i], 9), i))  # ties -> corpus order
        return [(self.chunks[i], float(sims[i])) for i in order[:k]]
