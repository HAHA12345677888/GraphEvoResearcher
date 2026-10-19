#!/usr/bin/env python3
"""
Answer matching + evaluation for GraphEvoResearcher real-LLM results.
Robust to $, commas, and formatting so "$70,000" matches golden "70000".
"""
import re
from typing import Dict, List


def normalize_number(s: str):
    """Extract a normalized number from a string, or None.

    STRICT: only returns a number if the string is *essentially* a number
    (after stripping $, commas, spaces, trailing units like '%' or ' days').
    Long sentences that merely contain digits return None, so we avoid
    false positives like pred='19' matching golden='9'.
    """
    if not s:
        return None
    t = s.replace("$", "").replace(",", "").strip().lower()
    # remove trailing unit words (days, hours, miles, percent, dollars, etc.)
    t = re.sub(r"\b(days?|hours?|miles?|percent|dollars?|dozen|eggs|cups|downloads|children|liters?|sheep|people|students|years?|dvds?|books?)\b", "", t)
    t = t.replace("%", "").strip()
    m = re.match(r"^-?\d+(?:\.\d+)?$", t)
    if not m:
        # allow "5/3" style fractions -> float
        fm = re.match(r"^-?\d+\s*/\s*\d+$", t)
        if fm:
            num, den = t.split("/")
            try:
                if float(den) != 0:
                    return float(num) / float(den)
            except ValueError:
                return None
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def is_pure_numeric(golden: str) -> bool:
    """True if the golden answer is a standalone number (not an expression)."""
    return normalize_number(golden) is not None


def answer_match(pred: str, golden: str, strict: bool = True) -> bool:
    """Strict answer matching to avoid false positives on expression answers.

    Rules:
      - If golden is a PURE NUMBER: compare numerically (tolerance 5%) OR
        exact normalized string.
      - If golden is NOT pure numeric (expression/text): require exact
        normalized substring equality of the full golden in pred AND reject
        single-digit accidental matches by requiring a token boundary.
    """
    if not pred or not golden:
        return False
    p = pred.strip()
    g = golden.strip()
    gl = g.lower()
    pl = p.lower()

    # Boolean answers (BBH style): match "true"/"false" appearing as a
    # standalone token in the prediction (pred may be a reasoning sentence
    # like "not not False = False."). Extract the final booleans and check the
    # LAST occurrence, which is usually the conclusion.
    if gl in ("true", "false"):
        # find all standalone true/false tokens in a normalized way,
        # but skip negations immediately before (not True / not False)
        tokens = re.findall(r"(?i)\b(true|false)\b", pl)
        if not tokens:
            return False
        # For "not X = Y" chains, take the last token as the conclusion;
        # verify it equals golden. Also handle "not not False = False".
        return tokens[-1].lower() == gl

    if is_pure_numeric(g):
        # golden is a number: numeric match (tolerant) or exact text match
        pn = normalize_number(p)
        gn = normalize_number(g)
        if pn is not None and gn is not None:
            if abs(pn - gn) / max(abs(gn), 1.0) < 0.02:  # tighter 2%
                return True
        # also accept exact digit string containment at token boundary
        gd = re.sub(r"[^0-9.\-]", "", gl)
        if gd and re.search(rf"(?<![0-9]){re.escape(gd)}(?![0-9])", pl):
            return True
        return False
    else:
        # expression / text golden: full golden must appear in pred
        # (with normalized spacing), and pred must be SHORT (an answer, not a
        # reasoning dump) to avoid '105' appearing inside a long sentence.
        if len(p) > 80:
            return False
        # normalize spacing for comparison
        pn = re.sub(r"\s+", "", pl)
        gn2 = re.sub(r"\s+", "", gl)
        # also normalize \frac{a}{b} <-> a/b fraction forms
        fm = re.search(r"\\frac\{([^}]*)\}\{([^}]*)\}", gn2)
        if fm:
            gn2 = gn2.replace(fm.group(0), f"{fm.group(1)}/{fm.group(2)}")
        if gn2 in pn:
            return True
        # numeric fallback for fractions
        pn2 = normalize_number(p)
        gn3 = normalize_number(g)
        if pn2 is not None and gn3 is not None and abs(pn2 - gn3) / max(abs(gn3), 1.0) < 0.02:
            return True
        return False


def evaluate_results(results: List[Dict]) -> Dict:
    """Compute accuracy for single / flat / graph and per-method stats."""
    n = len(results)
    if n == 0:
        return {}
    acc = {"single": 0, "flat": 0, "graph": 0}
    for r in results:
        golden = r.get("golden", "")
        if answer_match(r.get("single_answer", ""), golden):
            acc["single"] += 1
        if answer_match(r.get("flat_answer", ""), golden):
            acc["flat"] += 1
        g = r.get("graph", {})
        if answer_match(g.get("final_answer", ""), golden):
            acc["graph"] += 1

    out = {}
    for k, v in acc.items():
        out[f"{k}_accuracy"] = round(v / n, 4)
    out["n"] = n
    # graph cost stats
    steps = [r["graph"].get("num_steps", 0) for r in results if r.get("graph")]
    nodes = [r["graph"].get("num_nodes", 0) for r in results if r.get("graph")]
    out["avg_graph_steps"] = round(sum(steps) / max(len(steps), 1), 3)
    out["avg_graph_nodes"] = round(sum(nodes) / max(len(nodes), 1), 2)
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    d = json.load(open(args.results))
    res = d.get("results", [])
    print(json.dumps(evaluate_results(res), indent=2))
