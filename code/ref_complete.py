#!/usr/bin/env python3
"""Add a few missing core references via arXiv, then finalize latex_refs.json."""
import json
import re
import time
import urllib.parse
import urllib.request


def q(title):
    u = ("http://export.arxiv.org/api/query?search_query="
         + urllib.parse.quote("ti:\\\"" + title + "\\\"")
         + "&max_results=1")
    with urllib.request.urlopen(u, timeout=30) as r:
        return r.read().decode()


def parse(xml):
    t = re.search(r"<entry>.*?<title>([^<]+)</title>", xml, re.DOTALL)
    ids = re.findall(r"<id>([^<]+)</id>", xml)
    abs_id = next((i.split("/abs/")[-1] for i in ids if "/abs/" in i), None)
    year = re.findall(r"<published>([^<]+)</published>", xml)
    return (t.group(1).strip() if t else None,
            abs_id,
            (year[0][:4] if year else None))


def main():
    extra = [
        "2WikiMultiHopQA: Towards Multi-Hop Question Answering with Large Scale Multi-Hop Reasoning",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "Language Models are Few-Shot Learners",
        "Chain-of-Verification Reduces Hallucination in Large Language Models",
    ]
    cur = json.load(open("../latex_refs.json"))
    titles = {c["title"].lower() for c in cur}
    for ref in extra:
        for attempt in range(3):
            try:
                t, aid, yr = parse(q(ref))
                if aid and t and ref.lower()[:22] in t.lower():
                    if t.lower() not in titles:
                        cur.append({"key": f"ref{len(cur)+1}",
                                    "text": f"``{t},'' , {yr}. arXiv:{aid}",
                                    "title": t, "year": yr, "venue": "",
                                    "arxiv": aid, "source": "arxiv-extra"})
                        titles.add(t.lower())
                        print(f"[OK] {ref[:44]} -> {aid} ({yr})")
                    break
                print(f"[skip] {ref[:44]} -> {t[:40] if t else None}")
                break
            except Exception as e:
                print(f"[err] {ref[:40]} {type(e).__name__} {str(e)[:36]}; retry")
                time.sleep(6)
        time.sleep(1.2)
    print(f"\ntotal now: {len(cur)}")
    json.dump(cur, open("../latex_refs.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
