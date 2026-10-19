#!/usr/bin/env python3
"""Verify additional core references via the arXiv API (exact title match).
These are papers with arXiv as their primary published source that CrossRef
could not resolve reliably. Outputs arxiv_refs.json with real arXiv IDs.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

REFS = [
    "Training Verifiers to Solve Math Word Problems",          # GSM8K
    "Measuring Mathematical Problem Solving With the MATH Dataset",
    "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
    "Large Language Models Cannot Self-Correct Reasoning Yet",
    "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing",
    "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph",
    "Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning",
    "Inductive Representation Learning on Large Graphs",        # GraphSAGE
    "How Powerful are Graph Neural Networks?",                  # GIN
    "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering",
    "2WikiMultiHopQA: Towards Multi-Hop Question Answering with Large Scale Multi-Hop Reasoning",
    "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models",
    "Compositional Zero-Shot Learning with Classifier Alignment",
    "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks",
    "Large Language Models Are Few-Shot Learners",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "Attention Is All You Need",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)",
    "ReAct: Synergizing Reasoning and Acting in Language Models",
    "SELF-ALIGNMENT with Instruction Backtranslation",
]

def query(title):
    u = ("http://export.arxiv.org/api/query?search_query="
         + urllib.parse.quote("ti:\"" + title + "\"")
         + "&max_results=1")
    req = urllib.request.Request(u, headers={"User-Agent": "GraphEvoResearch/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()

def parse(xml):
    title = re.search(r"<entry>.*?<title>([^<]+)</title>", xml, re.DOTALL)
    idm = re.findall(r"<id>([^<]+)</id>", xml)
    pub = re.findall(r"<published>([^<]+)</published>", xml)
    authors = re.findall(r"<name>([^<]+)</name>", xml)
    # the entry's own id is the arxiv abs URL (e.g. http://arxiv.org/abs/2106.01761)
    abs_id = None
    for i in idm:
        if "/abs/" in i:
            abs_id = i.split("/abs/")[-1]
            break
    return {
        "title": title.group(1).strip() if title else None,
        "arxiv_id": abs_id,
        "year": pub[0][:4] if pub else None,
        "authors": authors,
    }

def main():
    out = []
    seen = set()
    for ref in REFS:
        ok = False
        for attempt in range(3):
            try:
                rec = parse(query(ref))
                if rec["arxiv_id"] and rec["title"]:
                    sim_ok = ref.lower()[:30] in rec["title"].lower() or rec["title"].lower()[:30] in ref.lower()
                    if sim_ok:
                        rec["wanted"] = ref
                        if rec["arxiv_id"] not in seen:
                            out.append(rec)
                            seen.add(rec["arxiv_id"])
                        print(f"[OK] {ref[:50]} -> {rec['arxiv_id']} ({rec['year']})")
                        ok = True
                        break
                    else:
                        print(f"[MISMATCH] {ref[:45]} -> {rec['title'][:45]}")
                        break
                else:
                    print(f"[NO-ENTRY] {ref[:45]}")
                    break
            except urllib.error.HTTPError as e:
                print(f"[HTTP {e.code}] {ref[:40]}; retry")
                time.sleep(8)
            except Exception as e:
                print(f"[ERR {type(e).__name__}] {ref[:40]}; retry")
                time.sleep(5)
            if ok:
                break
        time.sleep(1.5)
    print(f"\ntotal verified arXiv refs: {len(out)}")
    with open("../arxiv_refs.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
