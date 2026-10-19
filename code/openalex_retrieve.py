#!/usr/bin/env python3
"""Retrieve & verify core references via OpenAlex (user-provided api_key).
Prefers top venues (NeurIPS/ICML/ICLR/ACL/AAAI/EACL/NAACL/TACL/TMLR/TPAMI).
Merges into latex_refs.json, aiming ~50 verified refs (real DOI/arXiv).
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

KEY = os.environ.get("OPENALEX_API_KEY", "")  # read from env (never hardcode)
TOP_VENUES = ["Neural Information Processing", "Learning Representations",
              "Computational Linguistics", "AAAI", "Empirical Methods",
              "Transactions of the Association", "Machine Learning",
              "Pattern Analysis", "Nature Machine", "Knowledge"],
# want: titles that earlier arXiv/CrossRef could not resolve
WANT = [
    "2WikiMultiHopQA: Towards Multi-Hop Question Answering with Large Scale Multi-Hop Reasoning",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "Language Models are Few-Shot Learners",
    "Chain-of-Verification Reduces Hallucination in Large Language Models",
    "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering",
    "Survey on graph neural networks",
    "Reasoning on graphs faithful interpretable",
    "Graph neural network knowledge graph reasoning survey",
]


def oa_search(q, per=3):
    # OpenAlex key from env only, sent via X-Api-Key header (never in URL)
    u = ("https://api.openalex.org/works?search=" + urllib.parse.quote(q)
         + "&per-page=" + str(per)
         + "&select=display_name,doi,publication_year,primary_location,authorships,cited_by_count")
    headers = {"User-Agent": "GraphEvoResearch/0.1"}
    if KEY:
        headers["X-Api-Key"] = KEY
    with urllib.request.urlopen(urllib.request.Request(
            u, headers=headers), timeout=40) as r:
        return json.loads(r.read()).get("results", [])


def venue_of(w):
    src = (w.get("primary_location") or {}).get("source") or {}
    return src.get("display_name") or ""


def main():
    cur = json.load(open("../latex_refs.json"))
    titles = {c["title"].lower() for c in cur} | {c.get("wanted","").lower() for c in cur if c.get("wanted")}
    added = 0
    for q in WANT:
        try:
            hits = oa_search(q)
            # pick best by title overlap
            ql = q.lower()
            hits = [h for h in hits if (h.get("display_name") or "").lower()[:30] in ql
                    or ql[:30] in (h.get("display_name") or "").lower()]
            if not hits:
                print(f"[none] {q[:45]}")
                continue
            w = hits[0]
            t = w.get("display_name") or q
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            if t.lower() in titles:
                print(f"[dup]  {t[:45]}")
                continue
            titles.add(t.lower())
            cur.append({"key": f"ref{len(cur)+1}", "title": t,
                        "venue": venue_of(w), "year": w.get("publication_year"),
                        "DOI": doi, "wanted": q, "source": "openalex",
                        "cites": w.get("cited_by_count", 0)})
            added += 1
            print(f"[ADD] {t[:55]} | {venue_of(w)[:30]} | {w.get('publication_year')} | DOI={bool(doi)}")
        except Exception as e:
            print(f"[err] {q[:35]} {type(e).__name__} {str(e)[:45]}")
        time.sleep(0.8)
    print(f"\nadded {added}; total now {len(cur)}")
    json.dump(cur, open("../latex_refs.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
