#!/usr/bin/env python3
"""Assign semantic citekeys to named core papers, update latex_refs.json,
regenerate references.tex, and fix the body cite commands to use them."""
import json
import re

BASE = "/Users/liushuxing8888student.usm.my/paper/JKSU-CIS_GraphEvoResearcher"

# map: distinctive title fragment -> canonical citekey
SEMANTIC = [
    ("Chain-of-Thought Prompting Elicits Reasoning", "weicot2022"),
    ("Reflexion: language agents with verbal reinforcement", "reflexion2023"),
    ("Self-Refine: Iterative Refinement with Self-Feedback", "selfrefine2023"),
    ("Graph of Thoughts: Solving Elaborate Problems", "got2024"),
]


def main():
    p = f"{BASE}/latex_refs.json"
    refs = json.load(open(p))
    # assign canonical citekey to matched titles
    for r in refs:
        t = r.get("title", "").lower()
        for frag, key in SEMANTIC:
            if frag.lower() in t:
                r["citekey"] = key
                break
        else:
            r["citekey"] = r["key"]  # fallback to positional refN
    json.dump(refs, open(p, "w"), indent=2, ensure_ascii=False)
    # body cite fix: map old positional placeholders to canonical keys
    src = open(f"{BASE}/latex/skeleton.tex").read()
    # evoresearcher (our prior flat baseline) -> cite the reflection line + CoT
    body_map = {
        "weicot2022": "weicot2022",
        "reflexion2023": "reflexion2023",
        "selfrefine2023": "selfrefine2023",
        "got2024": "got2024",
    }
    # We previously mapped cot->ref11, reflexion->ref24, selfrefine->ref23,
    # got->ref36; replace those with the canonical keys.
    old_map = {"ref11": "weicot2022", "ref24": "reflexion2023",
               "ref23": "selfrefine2023", "ref36": "got2024"}
    def repl(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        out = []
        for k in keys:
            if k in old_map:
                out.append(old_map[k])
            elif k == "evoresearcher":
                out.append("selfrefine2023,reflexion2023")
            else:
                out.append(k)
        return "\\cite{" + ",".join(out) + "}"
    new, n = re.subn(r"\\cite\{([^}]+)\}", repl, src)
    open(f"{BASE}/latex/skeleton.tex", "w").write(new)
    print(f"body cite rewritten ({n} cite commands)")
    # regenerate references.tex from emit_refs-style logic but with citekey
    emit(f"{BASE}/latex_refs.json", f"{BASE}/latex/references.tex")
    print("references.tex regenerated")


def emit(refp, outp):
    refs = json.load(open(refp))
    lines = ["%%% automatically generated references %%%",
             "\\begin{thebibliography}{99}"]
    for r in refs:
        lines.append(f"\\bibitem{{{r.get('citekey','?')}}}")
        title = r.get("title", "").replace("&", "\\&").replace("_", "\\_")
        venue = r.get("venue", "") or ""
        year = r.get("year", "")
        doi = r.get("DOI", "")
        arx = r.get("arxiv_id", "")
        seg = " ".join([x for x in [f"{venue}, {year}" if venue else str(year) if year else ""] if x])
        body = f"\\emph{{{title}}}. {seg}."
        lines.append(body)
        loc = []
        if doi:
            loc.append(f"https://doi.org/{doi}")
        elif arx:
            loc.append(f"arXiv:{arx}")
        if loc:
            lines.append("\\newblock " + "; ".join(loc))
    lines.append("\\end{thebibliography}")
    open(outp, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
