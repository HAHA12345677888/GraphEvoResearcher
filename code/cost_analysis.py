#!/usr/bin/env python3
"""
Cost analysis for GraphEvoResearcher.
Graph protocol: graph_claims generations (1 per added node) + optional
final answer. Flat reflection: generation + critique + (possible) revision.
Single-shot: 1 generation.

We report RELATIVE generation counts and an estimated token cost using
a fixed per-call estimate (input ~700 tokens, output ~400). True token
counts can be added later from API usage fields.
"""
import json
import sys


def estimate_graph_calls(res):
    """Number of LLM generations for the graph protocol."""
    # graph: 1 per depth step + 1 for final/answer extraction
    return res["graph"].get("num_steps", 0) + 1


def estimate_flat_calls(_res):
    """Flat reflection = generate + critique (+1 revision)."""
    return 2  # generate + critique (revise only if needed)


def estimate_single_calls(_res):
    return 1


def analyze(path):
    d = json.load(open(path))
    results = d["results"]
    n = len(results)
    calls = {"single": 0, "flat": 0, "graph": 0}
    for r in results:
        calls["single"] += estimate_single_calls(r)
        calls["flat"] += estimate_flat_calls(r)
        calls["graph"] += estimate_graph_calls(r)
    # estimated tokens: input~700, output~400 per call
    per_call = 700 + 400
    tokens = {k: v * per_call for k, v in calls.items()}
    acc = {}
    # read accuracy by re-evaluating (reuse answer_match)
    sys.path.insert(0, "." if "." not in sys.path else sys.path)
    from evaluate import answer_match
    acc = {"single": 0, "flat": 0, "graph": 0}
    for r in results:
        g = r["golden"]
        if answer_match(str(r["single_answer"]), g): acc["single"] += 1
        if answer_match(str(r["flat_answer"]), g): acc["flat"] += 1
        if answer_match(str(r["graph"]["final_answer"]), g): acc["graph"] += 1
    print(f"Dataset: {path}  (n={n})")
    print(f"  {'method':<8}{'calls':>8}{'est_tokens':>12}{'accuracy':>10}{'acc/calls':>10}")
    for m in ["single", "flat", "graph"]:
        eff = round(acc[m] / n, 4)
        per = round(eff / calls[m], 4)
        print(f"  {m:<8}{calls[m]:>8}{tokens[m]:>12}{eff:>10.2%}{per:>10.4f}")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
