"""Ingestion-time metadata extraction (no LLM involved - plain regex, so it is auditable).

For every document we extract:
  title, effective date, kind (policy / memo / faq / other)
  supersedes   : "Supersedes the policy dated <date>"      -> version chain
  amends       : "clause <n> of the <Title>" in a memo       -> clause-level amendment link
and derive valid_from / valid_to for each document.
The memo in this corpus never repeats the words of the clause it amends ("shall not
exceed"), so lexical retrieval cannot connect them; the explicit clause reference is the
only reliable link, which is why it is extracted structurally.
"""
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from chunkers import clause_spans

DATE_RE = r"(\d{1,2} [A-Z][a-z]+ \d{4})"


def pdate(s):
    return datetime.strptime(s, "%d %B %Y").date()


@dataclass
class DocMeta:
    file: str
    title: str
    kind: str
    effective: date = None
    valid_to: date = None            # exclusive; set when a later doc supersedes it
    superseded_by: str = None
    supersedes: str = None
    amends: dict = field(default_factory=dict)    # {"file": f, "clause": "5.2"}
    amended_clauses: dict = field(default_factory=dict)   # clause -> memo file
    amended_spans: dict = field(default_factory=dict)     # clause -> (start,end)


def extract(docs):
    metas = {}
    for f, raw in docs.items():
        flat = re.sub(r"\s+", " ", raw)
        title = next(l.strip() for l in raw.splitlines() if l.strip())
        m = re.search(r"Effective " + DATE_RE, flat) or re.search(r"MEMO - " + DATE_RE, flat)
        kind = "memo" if "MEMO" in title else "faq" if "FREQUENTLY" in title else \
               "policy" if "POLICY" in title else "other"
        dm = DocMeta(f, title, kind, pdate(m.group(1)) if m else None)
        s = re.search(r"Supersedes the [a-z ]+dated " + DATE_RE, flat)
        if s:
            dm.supersedes = s.group(1)
        metas[f] = dm
    # version chains: same title + "supersedes <date>"
    for f, dm in metas.items():
        if dm.supersedes:
            old = [o for o in metas.values() if o.title == dm.title and o.effective == pdate(dm.supersedes)]
            for o in old:
                o.superseded_by, o.valid_to = f, dm.effective
    # clause-level amendments
    for f, dm in metas.items():
        if dm.kind != "memo":
            continue
        flat = re.sub(r"\s+", " ", docs[f])
        for cl, tgt in re.findall(r"clause (\d+\.\d+) of the (.+?) (?:is|are|has|shall)\b", flat):
            for o in metas.values():
                if o.title.lower() == tgt.lower():
                    dm.amends = {"file": o.file, "clause": cl}
                    o.amended_clauses[cl] = f
                    for no, s0, e0 in clause_spans(docs[o.file]):
                        if no == cl:
                            o.amended_spans[cl] = (s0, e0)
    return metas
