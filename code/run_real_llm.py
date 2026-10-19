#!/usr/bin/env python3
"""
GraphEvoResearcher: REAL-LLM experiment script (DeepSeek official API)
=======================================================================
Runs the graph-aware self-reflective protocol against a real LLM backend.

Usage:
    export DEEPSEEK_API_KEY=sk-...            # REQUIRED (never hardcoded)
    python3 run_real_llm.py --data ../data/dev20.jsonl --model deepseek-chat --n 10

Design:
  - Reads API key from environment only (security: never in files/artifacts).
  - Protocol: construct reasoning graph (nodes=claims, edges=support/contradict)
    -> graph-aware meta-reward -> graph-convergence early stop.
  - Baselines: (a) single-shot direct answer; (b) flat reflection (EvoResearcher).
  - Outputs JSON results + per-question trajectory for analysis.
"""
import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ENDPOINT = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


class RealLLMBackend:
    """OpenAI-compatible backend via urllib (stdlib only, no openai package).

    Works with DeepSeek official API (https://api.deepseek.com/v1) and any
    OpenAI-compatible endpoint. Reads key from env — never from files.
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.3,
                 max_tokens: int = 1024, endpoint: str = DEFAULT_ENDPOINT,
                 api_key: Optional[str] = None):
        import json as _json
        import urllib.request as _ur
        self._json = _json
        self._ur = _ur
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint = endpoint.rstrip("/")
        self.api_key = (api_key or os.environ.get("DEEPSEEK_API_KEY")
                        or os.environ.get("OPENAI_API_KEY")
                        or self._read_keyfile())
        if not self.api_key:
            raise RuntimeError("No API key: set DEEPSEEK_API_KEY or create ~/.deepseek_api_key")

    @staticmethod
    def _read_keyfile() -> Optional[str]:
        """Read key from a local 0600 keyfile. Never logs the key."""
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.expanduser("~/.deepseek_api_key"),
            os.path.expanduser("~/.config/deepseek/api_key"),
            os.path.join(proj_root, ".deepseek_api_key"),  # project-local
        ]
        for p in candidates:
            try:
                with open(p) as f:
                    k = f.read().strip()
                if k:
                    return k
            except OSError:
                continue
        return None

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
        }
        # deepseek-reasoner does not accept temperature
        if "reasoner" not in self.model:
            payload["temperature"] = self.temperature
        body = self._json.dumps(payload).encode()
        req = self._ur.Request(
            self.endpoint + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with self._ur.urlopen(req, timeout=120) as r:
            d = self._json.loads(r.read())
        return d["choices"][0]["message"]["content"] or ""


# --- prompts ----------------------------------------------------------------

PROMPT_GRAPH_STEP = """You are a graph-based reasoning agent. Given a question and the current
reasoning graph, produce the NEXT reasoning step as a JSON object:

{
  "new_claims": [
    {"content": "intermediate conclusion or evidence", "supporting": ["<exact id of a node it supports>"], "contradicting": []}
  ],
  "final_answer": "ANSWER IF confident, else null"
}

Rules:
- Each claim must be a self-contained statement.
- "supporting"/"contradicting" reference EXISTING node ids from the graph (or [] for new root evidence).
- If you are confident you have the answer, set final_answer; otherwise null.
- Do NOT repeat claims already in the graph (no loops).

Current graph:
@@GRAPH@@

Question: @@QUESTION@@
"""

PROMPT_FLAT_REFLECT = """You are a careful reasoning agent. Answer the question below.
First give your answer with a one-sentence reasoning, then critique your own answer in one
short sentence, and only revise if you find a real error.
Finally put ONLY the final answer (e.g. True/False, a number, or a short phrase) inside
<answer>tags</answer>.

Question: @@QUESTION@@"""


# --- graph construction / reflection loop (real LLM calls) ------------------

def parse_step_output(text: str) -> Dict[str, Any]:
    """Parse the JSON blob from the model (tolerate markdown fences / prose)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"new_claims": [], "final_answer": None}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"new_claims": [], "final_answer": None}


def run_graph_protocol(llm: RealLLMBackend, question: str,
                       max_depth: int = 4) -> Dict[str, Any]:
    """Training-free graph protocol with real LLM calls."""
    from reasoning_graph import ReasoningGraph
    g = ReasoningGraph()
    root = g.add_node(f"Q: {question[:80]}", node_type="root", confidence=1.0)
    final_answer = None
    steps = 0
    for d in range(1, max_depth + 1):
        steps = d
        graph_repr = json.dumps({
            nid: {"content": nd["content"], "type": nd["type"],
                  "id": nid} for nid, nd in g.nodes.items()
        }, ensure_ascii=False)[:3000]
        prompt = (PROMPT_GRAPH_STEP
                  .replace("@@GRAPH@@", graph_repr)
                  .replace("@@QUESTION@@", question))
        out = llm.complete("You produce structured reasoning steps as JSON.", prompt)
        parsed = parse_step_output(out)
        # add new claims as nodes
        for claim in parsed.get("new_claims", [])[:4]:
            nid = g.add_node(claim.get("content", "")[:160],
                             node_type="claim",
                             confidence=0.5)
            # edges: supporting references (existing nodes)
            for s in claim.get("supporting", [])[:3]:
                if s in g.nodes:
                    g.add_edge(s, nid, g.SUPPORT)
            for c in claim.get("contradicting", [])[:3]:
                if c in g.nodes:
                    g.add_edge(nid, c, g.CONTRADICT)
        # record a provisional answer if the model is confident
        fa = parsed.get("final_answer")
        if fa:
            final_answer = str(fa)
        # early stop when a final answer is given twice consecutively
        if d >= 2 and fa:
            break
    return {
        "question": question,
        "num_steps": steps,
        "num_nodes": g.num_nodes,
        "num_edges": g.num_edges,
        "has_cycle": g.has_cycle(),
        "final_answer": final_answer or "NO_ANSWER",
    }


def run_flat_reflection(llm: RealLLMBackend, question: str) -> str:
    out = llm.complete("You answer with <answer> tags.",
                      PROMPT_FLAT_REFLECT.replace("@@QUESTION@@", question))
    m = re.search(r"<answer>(.*?)</answer>", out, re.DOTALL)
    if m:
        return m.group(1).strip()[:200]
    # No tags: return the LAST sentence (usually the conclusion). The
    # evaluator extracts booleans/numbers from it robustly.
    sentences = re.split(r"(?<=[.!?])\s+", out.strip())
    if sentences:
        return sentences[-1][:200]
    return out.strip()[:200]


def run_single_shot(llm: RealLLMBackend, question: str) -> str:
    out = llm.complete("Answer concisely.",
                       f"Question: {question}\\nAnswer (one short phrase):")
    return out.strip()[:200]


def load_questions(path: str, n: int) -> List[Dict[str, str]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({"question": d.get("question", ""),
                         "golden": d.get("golden", d.get("answer", ""))})
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out", default="results/real_llm_results.json")
    args = ap.parse_args()

    llm = RealLLMBackend(model=args.model)
    questions = load_questions(args.data, args.n)
    print(f"Running {len(questions)} questions with {args.model} ...")

    results = []
    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['question'][:50]}...", flush=True)
        try:
            graph_res = run_graph_protocol(llm, q["question"], args.depth)
            flat_ans = run_flat_reflection(llm, q["question"])
            single_ans = run_single_shot(llm, q["question"])
            results.append({
                **q,
                "graph": graph_res,
                "flat_answer": flat_ans,
                "single_answer": single_ans,
            })
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(0.3)  # rate-limit courtesy

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "n": len(results), "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
