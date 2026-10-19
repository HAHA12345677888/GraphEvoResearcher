#!/usr/bin/env python3
"""Emit references.tex (a thebibliography block) from latex_refs.json."""
import json


def clean_title(t):
    # escape latex-sensitive chars lightly
    t = t.replace("&", "\\&").replace("#", "\\#").replace("_", "\\_")
    return t


def main():
    refs = json.load(open("../latex_refs.json"))
    lines = []
    lines.append("%%% automatically generated references %%%")
    lines.append("\\begin{thebibliography}{99}")
    for i, r in enumerate(refs, 1):
        lines.append(f"\\bibitem{{ref{i}}}")
        title = clean_title(r.get("title", ""))
        venue = r.get("venue", "") or ""
        year = r.get("year", "")
        doi = r.get("DOI", "")
        arx = r.get("arxiv_id", "")
        venue_lines = []
        if venue:
            venue_lines.append(f"{venue}, {year}")
        elif year:
            venue_lines.append(f"{year}")
        loc = []
        if doi:
            loc.append(f"https://doi.org/{doi}")
        elif arx:
            loc.append(f"arXiv:{arx}")
        body = f"\\emph{{{title}}}."
        joined = " ".join(venue_lines).strip()
        lines.append(f"{body} {joined}." if joined else f"{body}")
        if loc:
            lines.append("\\newblock " + "; ".join(loc))
    lines.append("\\end{thebibliography}")
    with open("../latex/references.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(refs)} bibitems to latex/references.tex")


if __name__ == "__main__":
    main()
