"""
Baseline: EvoResearcher-style flat reflection (training-free)
================================================================
Mirrors the EvoResearcher protocol: generate -> self-critique -> revise,
terminated by a CONFIRMED sentinel (implicit early stop) or max depth.

Used as the comparison baseline in the GraphEvoResearcher ablation.
Pure offline simulation (no API), consistent with the EvoResearcher
preliminary validation methodology.
"""

import random
from typing import Any, Dict, List, Optional


class EvoBaselineAgent:
    """Flat generate->critique->revise loop with CONFIRMED early stop."""

    def __init__(self, max_depth: int = 5, seed: int = 42,
                 confirm_prob: float = 0.6) -> None:
        self.max_depth = max_depth
        self.confirm_prob = confirm_prob
        self.rng = random.Random(seed)

    def _simulate_generation(self, question: str, golden: str,
                             difficulty: float, depth: int) -> Dict[str, Any]:
        """Simulate a generate step. Harder -> lower chance of landing correct
        on a given attempt; later depth -> slightly higher chance (refinement)."""
        # difficulty is the CALIBRATED single-shot accuracy (P correct first try)
        base_correct_prob = max(0.05, min(0.98, difficulty))
        # Reflection gain scales inversely with base accuracy: the loop helps
        # most where single-shot reasoning is unreliable (mirrors real finding:
        # MATH +14pp, GSM8K +4pp). Hard -> larger per-step gain.
        if base_correct_prob < 0.5:        # hard (MATH-like)
            refine = 0.07 * (depth - 1)
        elif base_correct_prob < 0.85:     # medium (BBH-like)
            refine = 0.03 * (depth - 1)
        else:                              # easy (GSM8K-like)
            refine = 0.01 * (depth - 1)
        p_correct = min(0.95, base_correct_prob + refine)
        correct = self.rng.random() < p_correct
        return {
            "answer": f"The answer is {golden}." if correct
                      else f"The answer is approximately {self.rng.choice(['unknown','unclear','needs more research'])}.",
            "correct": correct,
        }

    def run(self, question: str, golden: str,
            difficulty: float = 0.5) -> Dict[str, Any]:
        """Run the flat reflection loop. Returns trajectory + stop info."""
        steps = 0
        final_answer = ""
        correct = False
        early_stopped = False
        stop_step = 0

        for d in range(1, self.max_depth + 1):
            steps = d
            gen = self._simulate_generation(question, golden, difficulty, d)
            final_answer = gen["answer"]
            correct = gen["correct"]

            # CONFIRMED sentinel early stop (critique confirms the answer)
            if self.rng.random() < self.confirm_prob:
                early_stopped = True
                stop_step = d
                break

        return {
            "question": question,
            "golden": golden,
            "difficulty": difficulty,
            "final_answer": final_answer,
            "correct": correct,
            "early_stopped": early_stopped,
            "stop_step": stop_step,
            "depth_used": stop_step if early_stopped else self.max_depth,
            "num_steps": steps,
        }


def run_baseline_sweep(questions: List[Dict[str, Any]], max_depth: int = 5,
                       seed: int = 42) -> Dict[str, float]:
    """Run baseline across a question set (offline simulation)."""
    agent = EvoBaselineAgent(max_depth=max_depth, seed=seed)
    results = [agent.run(q["question"], q["golden"], q.get("difficulty", 0.5))
               for q in questions]
    n = len(results)
    acc = sum(1 for r in results if r["correct"]) / max(n, 1)
    early_stop = sum(1 for r in results if r["early_stopped"]) / max(n, 1)
    avg_depth = sum(r["depth_used"] for r in results) / max(n, 1)
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "early_stop_rate": round(early_stop, 4),
        "avg_depth_used": round(avg_depth, 3),
        "config": {"max_depth": max_depth, "seed": seed, "protocol": "flat-CONFIRMED"},
    }
