#!/usr/bin/env python3
"""
Preliminary ablation: GraphEvoResearcher vs EvoResearcher (plain reflection)
================================================================================
Compares on the SAME local benchmark questions (GSM8K / MATH / BBH subsets):

  Baseline (EvoResearcher-style):
    flat generate -> critique -> revise loop, CONFIRMED early stop,
    outcome-oriented meta-reward (correctness/efficiency/reflection/diversity)

  Ours (GraphEvoResearcher):
    explicit reasoning graph construction, graph-aware meta-reward
    (connectivity / cycle / branch / convergence), graph-convergence early stop

Metrics:
  accuracy, early-stop rate, avg depth used, plus graph-structure diagnostics
  (connectivity, cycle rate, branch coverage, convergence) to test H1/H2/H3.

This is an OFFLINE simulation (no API calls), mirroring the EvoResearcher
preliminary validation methodology, but over graph structures.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasoning_graph import (  # noqa: E402
    GraphEvoResearcher,
    GraphAwareMetaReward,
    ReasoningGraph,
    run_offline_sweep,
)
from baseline_reflection import (  # noqa: E402
    EvoBaselineAgent,
    run_baseline_sweep,
)


def load_local_benchmarks(data_dir: str, n_per_set: int = 50) -> dict:
    """Load local JSONL benchmarks (id, task, question, golden, type)."""
    sets = {}
    for name in ["gsm8k_500", "math_500", "bbh_500"]:
        path = os.path.join(data_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        # deterministic sample
        rows = rows[:n_per_set]
        sets[name] = rows
        print(f"  loaded {name}: {len(rows)} questions")
    return sets


def difficulty_from_task(task: str) -> float:
    """Return the CALIBRATED single-shot accuracy (0..1) for this benchmark,
    drawn from EvoResearcher's real crossbench results (deepseek-v4-flash, n=100):
        GSM8K single-shot 0.94, MATH 0.26, BBH 0.76.
    The simulator uses this as P(correct | single generation).

    NOTE: higher accuracy => EASIER. Downstream code treats this as
    'base_accuracy' (probability of getting it right first try), not a
    0..1 difficulty score.
    """
    task_l = task.lower()
    if "math" in task_l:
        return 0.26
    if "gsm" in task_l:
        return 0.94
    if "bbh" in task_l:
        return 0.76
    return 0.5


def run_ablation(data_dir: str, n_per_set: int = 50, max_depth: int = 5,
                 seed: int = 42) -> dict:
    print("=" * 70)
    print("GraphEvoResearcher preliminary ablation (offline simulation)")
    print("=" * 70)

    sets = load_local_benchmarks(data_dir, n_per_set)

    all_results = {}
    for set_name, rows in sets.items():
        print(f"\n--- {set_name} ({len(rows)} questions) ---")
        questions = [
            {"question": r["question"], "golden": r["golden"],
             "difficulty": difficulty_from_task(r.get("task", ""))}
            for r in rows
        ]

        # --- Baseline: EvoResearcher-style flat reflection ---
        baseline = run_baseline_sweep(questions, max_depth=max_depth, seed=seed)

        # --- Ours: GraphEvoResearcher ---
        ours = run_offline_sweep(questions, max_depth=max_depth,
                                 stop_threshold=0.75, seed=seed)

        all_results[set_name] = {
            "baseline": baseline,
            "graph": ours,
            "delta_accuracy_pp": round((ours["accuracy"] - baseline["accuracy"]) * 100, 2),
            "delta_early_stop_pp": round((ours["early_stop_rate"] - baseline["early_stop_rate"]) * 100, 2),
            "delta_avg_depth": round(ours["avg_depth_used"] - baseline["avg_depth_used"], 3),
        }

        print(f"  Baseline: acc={baseline['accuracy']:.3f}  early_stop={baseline['early_stop_rate']:.3f}  avg_depth={baseline['avg_depth_used']:.2f}")
        print(f"  Graph   : acc={ours['accuracy']:.3f}  early_stop={ours['early_stop_rate']:.3f}  avg_depth={ours['avg_depth_used']:.2f}  nodes={ours['avg_nodes']:.1f}")
        print(f"  Delta   : acc={all_results[set_name]['delta_accuracy_pp']:+.2f}pp  early_stop={all_results[set_name]['delta_early_stop_pp']:+.2f}pp  depth={all_results[set_name]['delta_avg_depth']:+.2f}")

    return {"results": all_results, "config": {
        "n_per_set": n_per_set, "max_depth": max_depth, "seed": seed,
        "mode": "offline-simulation", "note": "preliminary; real-LLM runs follow"
    }}


if __name__ == "__main__":
    import argparse
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(here, "..", "..", "github",
                                "plos one-EvoResearcher", "data")
    # resolve to the actual local path when run from the repo folder
    if not os.path.isdir(default_data):
        default_data = os.path.expanduser(
            "/Users/liushuxing8888student.usm.my/github/plos one-EvoResearcher/data")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=default_data)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join(here, "..", "results",
                                                      "preliminary_ablation.json"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result = run_ablation(args.data, n_per_set=args.n, max_depth=args.depth,
                          seed=args.seed)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.out}")
