#!/usr/bin/env python3
"""
Literature retrieval for the GraphEvoResearcher paper.
Collects REAL, recent (>=2021) references from top venues via CrossRef
(no API key required). Outputs a deduplicated JSON list.

Discipline: only real DOIs returned by the API are kept; nothing is
fabricated. Venue filtering uses the container-title match.
"""
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "GraphEvoResearchProject/0.1 (scholarly retrieval for a bibliography)"
TOPICS = [
    ("chain of thought reasoning", "Neural Information Processing"),
    ("chain of thought reasoning", "Learning Representations"),
    ("chain of thought prompting", "Annual Meeting of the Association for Computational Linguistics"),
    ("self-refine feedback", "Association for Computational Linguistics"),
    ("self-reflection agent", "Neural Information Processing"),
    ("reflexion verbal reinforcement", "Neural Information Processing"),
    ("graph of thoughts", "Neural Information Processing"),
    ("graph reasoning language model", "Learning Representations"),
    ("multi-hop question answering", "Association for Computational Linguistics"),
    ("multi-hop question answering", "Empirical Methods in Natural Language Processing"),
    ("knowledge graph reasoning network", "Learning Representations"),
    ("graph neural network reasoning", "Neural Information Processing"),
    ("efficient reasoning decomposition", "Association for Computational Linguistics"),
    ("large language model survey", "Neural Information Processing"),
    ("evaluation reasoning benchmark", "Association for Computational Linguistics"),
]


def fetch(query, container, rows=4):
    u = ("https://api.crossref.org/works?query.bibliographic="
         + urllib.parse.quote(query)
         + "&query.container-title=" + urllib.parse.quote(container)
         + "&filter=from-pub-date:2021-01-01,type:proceedings-article"
         + "&rows=" + str(rows)
         + "&select=title,DOI,published,container-title,author,is-referenced-by-count,DOI,publisher")
    req = urllib.request.Request(u, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read())["message"]["items"]


def main():
    seen = {}
    total = 0
    for topic, container in TOPICS:
        for attempt in range(4):
            try:
                items = fetch(topic, container)
                for it in items:
                    doi = it.get("DOI")
                    if not doi:
                        continue
                    seen.setdefault(doi, {
                        "title": (it.get("title") or ["?"])[0],
                        "venue": (it.get("container-title") or ["?"])[0],
                        "year": (it.get("published", {}).get("date-parts", [[None]])[0][0]),
                        "authors": [a.get("family", "") for a in (it.get("author") or [])][:6],
                        "cites": it.get("is-referenced-by-count", 0),
                    })
                    total += 1
                print(f"OK  [{topic} | {container}] -> {len(items)}")
                break
            except urllib.error.HTTPError as e:
                print(f"HTTP {e.code} [{topic}] attempt {attempt}; wait")
                time.sleep(8)
            except Exception as e:
                print(f"FAIL {type(e).__name__} [{topic}]; wait")
                time.sleep(6)
        time.sleep(2.5)

    print(f"\nTOTAL unique DOIs: {len(seen)}")
    with open("../literature.json", "w") as f:
        json.dump(list(seen.values()), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
