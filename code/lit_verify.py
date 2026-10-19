#!/usr/bin/env python3
"""
Literature verification v2: query CrossRef by title, keep only high-similarity
matches, and store DOIs. Outputs bibliography.json with verified entries.

Every entry carries a real DOI returned by CrossRef; nothing is hand-typed
from memory. Entries with low title similarity are flagged, not saved.
"""
import difflib
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

socket.setdefaulttimeout(30)
UA = "GraphEvoResearchProject/0.1 (bibliography verification)"

# Core references by chapter (title + expected venue keyword for sanity)
REFS = [
    # CoT / reasoning
    ("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "Neural Information Processing"),
    ("Large Language Models are Zero-Shot Reasoners", "Neural Information Processing"),
    ("Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "Neural Information Processing"),
    ("Self-Consistency Improves Chain of Thought Reasoning in Language Models", "Learning Representations"),
    ("Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them", "Learning Representations"),
    ("Training Verifiers to Solve Math Word Problems", "Computational Linguistics"),
    ("Measuring Mathematical Problem Solving With the MATH Dataset", "Neural Information Processing"),
    ("Large Language Models Cannot Self-Correct Reasoning Yet", "Learning Representations"),
    # self-reflection / self-correction
    ("Self-Refine: Iterative Refinement with Self-Feedback", "Neural Information Processing"),
    ("Reflexion: Language agents with verbal reinforcement learning", "Neural Information Processing"),
    ("CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing", "Learning Representations"),
    ("Large Language Models are Better Reasoners with Self-Verification", "Findings"),
    # graph reasoning / GoT
    ("Graph of Thoughts: Solving Elaborate Problems with Large Language Models", "AAAI"),
    ("Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning", "AAAI"),
    ("Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph", "Learning Representations"),
    ("Reasoning with Language Model is Planning with World Model", "Empirical Methods"),
    # multi-hop QA
    ("MuSiQue: Multihop Questions via Single-hop Question Composition", "Association for Computational Linguistics"),
    ("2WikiMultiHopQA: Towards Multi-Hop Question Answering with Large Scale Multi-Hop Reasoning", ""),
    ("Multi-hop Question Answering via Reasoning Chains", ""),
    ("Improving Multi-hop Question Answering over Knowledge Graphs Using Knowledge Base Embeddings", ""),
    # graph neural networks
    ("Inductive Representation Learning on Large Graphs", "Neural Information Processing"),
    ("Graph Attention Networks", "Learning Representations"),
    ("How Powerful are Graph Neural Networks?", "Learning Representations"),
    ("Heterogeneous Graph Attention Network", "Web Search"),
    ("Dynamic Graph Neural Networks: A Survey", ""),
    # evaluation / benchmarks
    ("GSM8K: Training Verifiers to Solve Math Word Problems", ""),
    ("Chain-of-Thought Reasoning Without Prompting", "Neural Information Processing"),
    ("A Survey on Large Language Models: Applications, Challenges, Limitations", ""),
]


def norm(s):
    return " ".join(s.lower().split())


def title_similarity(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def fetch_title(title, rows=3):
    u = ("https://api.crossref.org/works?query.title=" + urllib.parse.quote(title)
         + "&rows=" + str(rows)
         + "&select=title,DOI,container-title,published,author,is-referenced-by-count")
    req = urllib.request.Request(u, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["message"]["items"]


def main():
    out = []
    seen = set()
    for want, venue_kw in REFS:
        ok = False
        for attempt in range(3):
            try:
                items = fetch_title(want)
                # pick best similarity
                best = max(items, key=lambda x: title_similarity(want, (x.get("title") or ["?"])[0]))
                sim = title_similarity(want, (best.get("title") or ["?"])[0])
                doi = best.get("DOI")
                if doi and sim >= 0.60 and doi not in seen:
                    rec = {
                        "wanted": want,
                        "title": (best.get("title") or ["?"])[0],
                        "DOI": doi,
                        "venue": (best.get("container-title") or ["?"])[0],
                        "year": (best.get("published") or {}).get("date-parts", [[None]])[0][0],
                        "authors": [a.get("family", "") for a in (best.get("author") or [])][:8],
                        "cites": best.get("is-referenced-by-count", 0),
                        "similarity": round(sim, 3),
                    }
                    out.append(rec)
                    seen.add(doi)
                    ok = True
                    print(f"[OK sim={sim:.2f}] {want[:55]} -> {rec['title'][:45]} | {rec['DOI'][:35]}")
                    break
                elif doi and sim < 0.60:
                    print(f"[LOW sim={sim:.2f}] {want[:55]} -> {(best.get('title') or ['?'])[0][:45]}")
                    break
                else:
                    print(f"[NO-DOI] {want[:55]}")
                    break
            except urllib.error.HTTPError as e:
                print(f"[HTTP {e.code}] {want[:40]}; retry")
                time.sleep(8)
            except Exception as e:
                print(f"[ERR {type(e).__name__}] {want[:40]}; retry")
                time.sleep(5)
            if ok:
                break
        time.sleep(1.5)

    print(f"\nverified entries with DOI: {len(out)}")
    with open("../bibliography.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
