"""Conflict / staleness resolution layer (the 'judgement call', Step 4).

Design: *effective-date metadata + provenance disclosure*.
  1. As-of date: parsed from the query ("June 2019"); default = today.
  2. Documents not in force on that date are removed (superseded / not yet effective).
  3. Clauses amended by a memo are redacted from the original chunk and the amending
     memo chunk is pulled in instead (even if retrieval never surfaced it).
  4. Every removal is reported back as a NOTE, so the user always sees which version
     governed and what was set aside - nothing is silently picked.
  5. If two *different* versions of the same document family are still in force and no
     metadata resolves them, we refuse to pick and surface both.
"""
import re
from datetime import date
from chunkers import Chunk

MONTHS = "january february march april may june july august september october november december".split()


def parse_as_of(query, today=None):
    today = today or date.today()
    q = query.lower()
    m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", q)
    if m:
        return date(int(m.group(2)), MONTHS.index(m.group(1)) + 1, 15), True
    m = re.search(r"\b(?:in|during|for|of)\s+((?:19|20)\d\d)\b", q)
    if m:
        return date(int(m.group(1)), 6, 30), True
    return today, False


def fmt(d):
    return d.strftime("%d %b %Y").lstrip("0")


def resolve(candidates, metas, docs, chunks_by_file, as_of, explicit_date, k=3):
    notes, kept = [], []   # notes are (text, related_files)
    seen_files_removed = set()
    for ch, sc in candidates:
        dm = metas[ch.file]
        if dm.effective and as_of < dm.effective:
            notes.append((f"{ch.file} (effective {fmt(dm.effective)}) was not yet in force on {fmt(as_of)}; ignored.", {ch.file}))
            continue
        if dm.valid_to and as_of >= dm.valid_to:
            seen_files_removed.add(ch.file)
            continue
        # amended clause? -> redact / swap in the amending memo
        swapped = False
        for cl, memo in dm.amended_clauses.items():
            if as_of < metas[memo].effective:
                continue
            a, b = dm.amended_spans[cl]
            if ch.end <= a or ch.start >= b:
                continue                                  # no overlap
            note = (f"clause {cl} of {ch.file} was amended by {memo} ({fmt(metas[memo].effective)}); "
                    f"the original wording of clause {cl} is no longer in force.")
            if (note, {ch.file, memo}) not in notes:
                notes.append((note, {ch.file, memo}))
            if a <= ch.start and ch.end <= b:             # chunk is entirely the amended clause
                for mch in chunks_by_file[memo]:
                    kept.append((mch, sc))
                swapped = True
            else:                                         # partial overlap -> redact that part
                lo, hi = max(a, ch.start) - ch.start, min(b, ch.end) - ch.start
                body = ch.body[:lo] + f"[clause {cl} redacted: amended by {memo}]" + ch.body[hi:]
                ch = Chunk(ch.cid, ch.file, ch.header, body, ch.start, ch.end, ch.strategy)
                for mch in chunks_by_file[memo]:
                    kept.append((mch, sc * 0.999))
        if not swapped:
            kept.append((ch, sc))
    for f in sorted(seen_files_removed):
        dm = metas[f]
        notes.append((f"{f} (effective {fmt(dm.effective)}) is superseded by {dm.superseded_by} "
                      f"(effective {fmt(metas[dm.superseded_by].effective)}); answer uses the version in force on {fmt(as_of)}.", {f, dm.superseded_by}))
    if explicit_date:
        newer = [d for d in metas.values() if d.supersedes and d.effective > as_of]
        for d in newer:
            notes.append((f"You asked about {fmt(as_of)}; a later version {d.file} (effective {fmt(d.effective)}) changed this policy.", {o.file for o in metas.values() if o.superseded_by == d.file}))
    # de-duplicate, keep best score
    best = {}
    for ch, sc in kept:
        if ch.cid not in best or sc > best[ch.cid][1]:
            best[ch.cid] = (ch, sc)
    final = sorted(best.values(), key=lambda x: -x[1])[:k]
    # unresolved conflict: >1 distinct in-force docs with the same title
    fam = {}
    for ch, _ in final:
        fam.setdefault(metas[ch.file].title, set()).add(ch.file)
    conflict = {t: sorted(fs) for t, fs in fam.items() if len(fs) > 1 and metas[next(iter(fs))].kind == "policy"}
    return final, notes, conflict
