#!/usr/bin/env python3
"""OpenAlex backfill: fetch DOI for venue-only refs. Reads key from env."""
import json
import os
import time
import urllib.parse
import urllib.request

KEY = os.environ.get("OPENALEX_API_KEY", "")


def oa_doi(title):
    if not KEY:
        return None
    # OpenAlex key is read ONLY from the OPENALEX_API_KEY env var (never hardcoded)
    # and sent via an Authorization header rather than the URL query string.
    try:
        u = ("https://api.openalex.org/works?search=" + urllib.parse.quote(title)
             + "&per-page=1&select=display_name,doi,publication_year")
        headers = {"User-Agent": "GraphEvoResearch/0.1"}
        if KEY:
            headers["X-Api-Key"] = KEY
        with urllib.request.urlopen(urllib.request.Request(
                u, headers=headers), timeout=40) as r:
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
    p = "/Users/liushuxing8888student.usm.my/paper/JKSU-CIS_GraphEvoResearcher/latex_refs.json"
    refs = json.load(open(p))
    filled = 0
    for r in refs:
        if r.get("has_id"):
            continue
        doi = oa_doi(r.get("title", ""))
        if doi:
            r["DOI"] = doi
            r["identifier"] = doi
            r["has_id"] = True
            filled += 1
            print(f"[DOI] {r['title'][:50]} -> {doi}")
        time.sleep(0.5)
    json.dump(refs, open(p, "w"), indent=2, ensure_ascii=False)
    n_id = sum(1 for r in refs if r.get("has_id"))
    print(f"backfilled {filled}; now {n_id}/{len(refs)} with identifier")


if __name__ == "__main__":
    main()
