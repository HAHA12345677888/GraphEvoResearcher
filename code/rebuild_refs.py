#!/usr/bin/env python3
"""Rebuild latex_refs.json cleanly from the raw verified sources:
  - arxiv_refs.json  (16 real arXiv IDs, primary source = arXiv)
  - final_refs.json  (30 entries, 14 with CrossRef DOI, rest with venue+year)
  - OpenAlex backfill for entries lacking a DOI (where an exact match exists)
Every entry keeps its identifier (arXiv id or DOI). Entries that end up with
only venue+year are kept but explicitly flagged (identifier=None) so the count
of *verified* refs is honest.
"""
import json
import time
import urllib.parse
import urllib.request

KEY = __import__("os").environ.get("OPENALEX_API_KEY", "")


def oa_doi(title):
    if not KEY:
        return None
    try:
        u = ("https://api.openalex.org/works?search=" + urllib.parse.quote(title)
             + "&per-page=1&api_key=" + KEY + "&select=display_name,doi,publication_year")
        with urllib.request.urlopen(urllib.request.Request(
                u, headers={"User-Agent": "GraphEvoResearch/0.1"}), timeout=40) as r:
            res = json.loads(r.read()).get("results", [])
    except Exception:
        return None
    if not res:
        return None
    w = res[0]
    got = (w.get("display_name") or "").lower()
    ql = title.lower()
    if got[:30] in ql or ql[:30] in got:
        return (w.get("doi") or "").replace("https://doi.org/", "")
    return None


def main():
    base = "/Users/liushuxing8888student.usm.my/paper/JKSU-CIS_GraphEvoResearcher"
    arv = json.load(open(f"{base}/arxiv_refs.json"))   # 16 with arxiv_id
    fin = json.load(open(f"{base}/final_refs.json"))   # 30 with some DOI

    by_title = {}
    for a in arv:
        by_title[a.get("title", "").lower()] = {
            "title": a.get("title", ""), "arxiv_id": a.get("arxiv_id"),
            "year": a.get("year"), "venue": a.get("venue", "") or "", "source": "arxiv",
        }
    for f in fin:
        t = f.get("title", "").lower()
        rec = by_title.get(t) or {"title": f.get("title", ""), "arxiv_id": None,
                                  "year": f.get("year"), "venue": f.get("venue", "") or "",
                                  "source": "crossref"}
        if f.get("DOI"):
            rec["DOI"] = f["DOI"]
            rec.setdefault("source", "crossref")
        by_title[t] = rec

    # OpenAlex backfill for entries with neither DOI nor arxiv_id
    noid = [r for r in by_title.values()
            if not r.get("DOI") and not r.get("arxiv_id")]
    filled = 0
    for r in noid:
        doi = oa_doi(r.get("title", ""))
        if doi:
            r["DOI"] = doi
            filled += 1
            print(f"[DOI] {r['title'][:48]} -> {doi}")
        time.sleep(0.5)
    print(f"openalex backfilled {filled}")

    refs = list(by_title.values())
    refs.sort(key=lambda r: (str(r.get("year")), r.get("title", "")))
    out = []
    for i, r in enumerate(refs, 1):
        r["key"] = f"ref{i}"
        r["identifier"] = r.get("DOI") or r.get("arxiv_id") or None
        r["has_id"] = bool(r["identifier"])
        out.append(r)
    with open(f"{base}/latex_refs.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    n_id = sum(1 for r in out if r["has_id"])
    print(f"RESULT: {len(out)} refs total, {n_id} with identifier (DOI/arXiv), {len(out)-n_id} venue-only")


if __name__ == "__main__":
    main()
