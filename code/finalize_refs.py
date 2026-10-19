#!/usr/bin/env python3
"""Back-fill DOIs via OpenAlex for refs lacking DOI/arXiv, then emit
references.tex (LaTeX bibliography). Every ref ends with a real DOI or arXiv id.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

KEY = os.environ.get("OPENALEX_API_KEY", "")  # read from env (never hardcode)


def oa_doi(title):
    try:
        u = ("https://api.openalex.org/works?search=" + urllib.parse.quote(title)
             + "&per-page=1&api_key=" + KEY
             + "&select=display_name,doi,publication_year")
        with urllib.request.urlopen(urllib.request.Request(
                u, headers={"User-Agent": "GraphEvoResearch/0.1"}), timeout=40) as r:
            res = json.loads(r.read()).get("results", [])
    except urllib.error.HTTPError as e:
        return None
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
    refs = json.load(open("../latex_refs.json"))
    fixed = 0
    for r in refs:
        text = r.get("text", "")
        if "DOI:" in text or "arXiv:" in text or r.get("arxiv_id") or r.get("arxiv"):
            continue
        doi = oa_doi(r.get("title", ""))
        if doi:
            r["DOI"] = doi
            fixed += 1
            print(f"[DOI] {r['title'][:48]} -> {doi}")
        else:
            print(f"[NONE] {r['title'][:48]}")
        time.sleep(0.6)
    print(f"\nback-filled {fixed} DOIs; total {len(refs)}")

    # also restore DOI from final_refs.json bib entries (14 CrossRef DOIs)
    try:
        fin = json.load(open("../final_refs.json"))
        for fr in fin:
            if not fr.get("DOI"):
                continue
            for r in refs:
                if r.get("title","").lower()[:45] == fr["title"].lower()[:45] and "DOI:" not in r.get("text",""):
                    r["DOI"] = fr["DOI"]
                    r["text"] = r.get("text","").replace("arXiv:","")[:0] or r.get("text","")
        print("restored CrossRef DOIs from final_refs.json")
    except Exception as e:
        print("final_refs merge skip:", e)

    json.dump(refs, open("../latex_refs.json", "w"), indent=2, ensure_ascii=False)
    loc_count = sum(1 for r in refs if "DOI:" in r.get("text","") or "arXiv:" in r.get("text","")
                    or r.get("arxiv_id") or r.get("arxiv") or r.get("DOI"))
    print("refs with DOI or arXiv now:", loc_count, "/", len(refs))


if __name__ == "__main__":
    main()
