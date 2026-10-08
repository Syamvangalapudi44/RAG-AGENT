"""Two genuinely different chunking strategies.

1. fixed_chunks     - blind sliding window over raw characters (size + overlap).
                      Knows nothing about clauses, tables or Q&A pairs.
2. structure_chunks - parses the document: one chunk per numbered clause,
                      per table row, per Q&A pair, or per paragraph/memo. Every
                      chunk gets a contextual header (document title + effective
                      date) so a clause is never separated from what it belongs to.
"""
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    cid: str
    file: str
    header: str      # context prefix (empty for fixed chunks)
    body: str        # the raw text the chunk covers
    start: int       # char span of the body in the raw file
    end: int
    strategy: str

    @property
    def text(self):  # what gets embedded / shown to the LLM
        return (self.header + "\n" + self.body).strip() if self.header else self.body


# ---------------------------------------------------------------- strategy 1
def fixed_chunks(docs, size=200, overlap=40):
    out = []
    for fname, raw in docs.items():
        step = size - overlap
        i, n = 0, 0
        while i < len(raw):
            piece = raw[i:i + size]
            if n > 0 and len(piece) <= overlap:   # tail fully inside previous chunk
                break
            out.append(Chunk(f"{fname}#f{n}", fname, "", piece, i, i + len(piece), "fixed"))
            n += 1
            i += step
    return out


# ---------------------------------------------------------------- strategy 2
CLAUSE_RE = re.compile(r"^\d+\.\d+\s", re.M)


def doc_header(raw):
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    title = lines[0]
    m = re.search(r"(Effective [^.\n]+\d{4})", raw) or re.search(r"(MEMO - [^\n]+)", raw)
    return title + (f" ({m.group(1)})" if m and m.group(1) not in title else "")


def clause_spans(raw):
    """[(clause_no, start, end)] for numbered clauses incl. wrapped continuation lines."""
    ms = list(CLAUSE_RE.finditer(raw))
    spans = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(raw)
        spans.append((m.group(0).strip(), m.start(), len(raw[:end].rstrip())))
    return spans


def structure_chunks(docs):
    out = []
    for fname, raw in docs.items():
        header = doc_header(raw)
        n = 0

        def add(body, s, e, hdr=header):
            nonlocal n
            out.append(Chunk(f"{fname}#s{n}", fname, hdr, body, s, e, "structure"))
            n += 1

        if "|---" in raw:                                   # table -> one chunk per row
            lines = raw.split("\n")
            pos, cols, intro = 0, None, ""
            for ln in lines:
                s, e = pos, pos + len(ln)
                pos = e + 1
                if ln.startswith("|"):
                    cells = [c.strip() for c in ln.strip("|").split("|")]
                    if cols is None:
                        cols = cells
                    elif set(ln.replace("|", "").strip()) <= {"-", " "}:
                        continue
                    else:
                        row = " | ".join(f"{c}: {v}" for c, v in zip(cols, cells))
                        add(f"{intro} {row}".strip(), s, e)
                elif ln.strip() and not ln.startswith("SEPARATION") and cols is None:
                    intro = ln.strip().rstrip(".")
                elif ln.strip() and cols is not None:        # trailing paragraph
                    add(ln.strip(), s, e)
        elif CLAUSE_RE.search(raw):                          # numbered clauses
            for no, s, e in clause_spans(raw):
                add(re.sub(r"\s+", " ", raw[s:e]), s, e)
        elif re.search(r"^Q:", raw, re.M):                   # FAQ -> one chunk per Q/A pair
            for m in re.finditer(r"^Q:.*?(?=^Q:|\Z)", raw, re.M | re.S):
                add(re.sub(r"\s+", " ", m.group(0)).strip(), m.start(), m.end())
        else:                                                # memo / prose -> whole body
            body_start = raw.find("Subject:")
            body_start = 0 if body_start < 0 else body_start
            add(re.sub(r"\s+", " ", raw[body_start:]).strip(), body_start, len(raw))
    return out


CHUNKERS = {"fixed": fixed_chunks, "structure": structure_chunks}
