"""Grounded answer generation.

LLM mode (set env vars, see README):
    LLM_PROVIDER = openai | ollama | anthropic     (openai = any OpenAI-compatible endpoint,
                                                    e.g. your portal's LLM proxy via LLM_BASE_URL)
    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
Offline mode (default when LLM_PROVIDER is unset): a deterministic *extractive* generator that
returns the best-matching clause(s) from the retrieved chunks. It is a stand-in so the pipeline
and the evaluation harness run anywhere; it is NOT a substitute for an LLM and the write-up says so.
"""
import json, os, re, urllib.request
from embedder import tokenize

ABSTAIN_TEXT = "I don't know - the documents I have do not contain this information."
ABSTAIN_RE = re.compile(r"i don't know|i do not know|do not contain|does not contain|no information|"
                        r"not (?:specified|mentioned|covered in the)|cannot (?:be )?(?:answer|determine)|insufficient", re.I)

SYSTEM = ("You answer questions about company policy using ONLY the provided context chunks. "
          "Cite the source file of every fact in square brackets, e.g. [travel_policy_2024.md]. "
          "If the context does not contain the answer, reply exactly: " + ABSTAIN_TEXT + " "
          "Do not use outside knowledge. Be concise (1-3 sentences). "
          "If the context carries NOTE lines about superseded or amended sources, answer from the "
          "governing source and add one final line starting with 'NOTE:' summarising the provenance.")


def build_prompt(query, chunks, notes):
    ctx = "\n\n".join(f"[{c.file}]\n{c.text}" for c, _ in chunks)
    n = ("\n\nNOTES FROM THE RESOLVER:\n" + "\n".join("- " + x for x in notes)) if notes else ""
    return f"CONTEXT:\n{ctx}{n}\n\nQUESTION: {query}"


def llm_available():
    return os.environ.get("LLM_PROVIDER", "").lower() in ("openai", "ollama", "anthropic")


def call_llm(prompt):
    p = os.environ["LLM_PROVIDER"].lower()
    model = os.environ.get("LLM_MODEL", {"openai": "gpt-4o-mini", "ollama": "llama3", "anthropic": "claude-haiku-4-5"}[p])
    key = os.environ.get("LLM_API_KEY", "")
    if p == "anthropic":
        req = urllib.request.Request(os.environ.get("LLM_BASE_URL", "https://api.anthropic.com") + "/v1/messages", method="POST",
            data=json.dumps({"model": model, "max_tokens": 300, "temperature": 0, "system": SYSTEM,
                             "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
        return json.load(urllib.request.urlopen(req, timeout=60))["content"][0]["text"].strip()
    base = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1" if p == "ollama" else "https://api.openai.com/v1")
    req = urllib.request.Request(base + "/chat/completions", method="POST",
        data=json.dumps({"model": model, "temperature": 0, "messages": [
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}).encode(),
        headers={"content-type": "application/json", "authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"].strip()


# ------------------------------------------------------------------ offline extractive stand-in
def units_of(body):
    if "|" in body and "\n" in body:
        return [l for l in body.split("\n") if l.strip()]
    flat = re.sub(r"\s+", " ", body).strip()
    parts = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9])", flat)
    return [p for p in parts if p.strip()]


def extractive(query, chunks, embedder, min_cover):
    q = set(tokenize(query))
    total = sum(embedder.idf_of(t) for t in q) or 1.0
    scored = []
    for rank, (c, s) in enumerate(chunks):
        for u in units_of(c.body):
            ut = set(tokenize(u))
            sc = sum(embedder.idf_of(t) for t in q if t in ut)
            if sc > 0:
                scored.append((sc - 0.01 * rank, u, c.file, sc / total))
    if not scored:
        return None, []
    scored.sort(key=lambda x: -x[0])
    best = scored[0]
    if best[3] < min_cover:
        return None, []
    picked = [best]
    for cand in scored[1:]:
        if len(picked) >= 3:
            break
        if cand[0] >= 0.85 * best[0] and cand[1] not in [p[1] for p in picked]:
            picked.append(cand)
    ans = " ".join(p[1] for p in picked)
    files = []
    for p in picked:
        if p[2] not in files:
            files.append(p[2])
    return ans, files


def generate(query, chunks, notes, embedder, min_cover=0.0):
    """returns (answer_text, cited_files, abstained)"""
    if llm_available():
        text = call_llm(build_prompt(query, chunks, notes))
        if ABSTAIN_RE.search(text) and not re.search(r"\d", text):
            return ABSTAIN_TEXT, [], True
        files = list(dict.fromkeys(re.findall(r"\[([\w\-]+\.md)\]", text))) or list(dict.fromkeys(c.file for c, _ in chunks))
        return text, files, False
    ans, files = extractive(query, chunks, embedder, min_cover)
    if ans is None:
        return ABSTAIN_TEXT, [], True
    return ans, files, False
