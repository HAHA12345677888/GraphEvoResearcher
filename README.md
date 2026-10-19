# GraphEvoResearcher

**Graph-Aware Training-Free Self-Reflective Reasoning for Cost-Bounded
Multi-Hop Question Answering**

Companion code for the manuscript submitted to the *Journal of King
Saud University -- Computer and Information Sciences* (JKSU-CIS),
special issue *"Graph Intelligence: Learning, Reasoning, and Network
Applications"*.

## Overview

GraphEvoResearcher lifts the flat self-reflective loop
(`generate -> critique -> revise`, as in EvoResearcher) onto an explicit
**reasoning graph**: nodes are intermediate conclusions/evidence and
edges encode `support`, `contradict`, and `depends_on`. A graph-aware
meta-reward combines correctness, efficiency, reflection, diversity and
four structural signals (connectivity, acyclicity, branch coverage, path
convergence) to decide whether to revise, backtrack, or stop. A
graph-convergence early-stop terminates when evidence branches converge
on a single conclusion.

The protocol is **training-free** and dataset-agnostic (no external
knowledge graph is required).

## Real-LLM results (DeepSeek official API, n=50 per dataset, strict match)

| Dataset | Single-shot | Flat reflection | **GraphEvoResearcher** |
|---|---|---|---|
| GSM8K | 76.0% | 96.0% | **98.0%** |
| MATH  | 68.0% | 80.0% | **92.0%** |
| BBH   | 74.0% | 80.0% | **82.0%** |

Cross-model verification (MATH, `deepseek-reasoner`, n=25): 92% →
92% → **96%**, confirming that graph-structural gains are largest where
single-shot reliability is lowest.

## Repository layout

```
code/
  reasoning_graph.py        # graph structure, graph-aware meta-reward,
                            # graph-convergence early stop, visualization
  baseline_reflection.py    # flat generate->critique->revise baseline
  run_preliminary_ablation.py  # offline simulation ablation
  run_real_llm.py           # real-LLM protocol (urllib, zero deps)
  evaluate.py               # strict answer matching + evaluation
  cost_analysis.py          # relative generation-cost analysis
  lit_retrieve.py           # bibliography retrieval (CrossRef)
```

## Quick start

```bash
# 1. Provide your DeepSeek API key via env (never commit the key)
export DEEPSEEK_API_KEY=sk-...

# 2. Run a small real-LLM comparison (example: 25 questions)
python3 code/run_real_llm.py \
    --data data/dev25_math.jsonl --n 25 --depth 3 \
    --model deepseek-chat --out results/real_math_25.json

# 3. Evaluate
python3 code/evaluate.py --results results/real_math_25.json
```

The key is read from `DEEPSEEK_API_KEY` (env), `~/.deepseek_api_key`,
or a project-local `.deepseek_api_key` (mode 0600). It is **never**
hard-coded in the codebase.

## Data

Data files (GSM8K/MATH/BBH subsets) are not committed here; prepare
JSONL lines `{"question": ..., "golden": ...}` and point `--data` at
your file, or use the original open datasets.

## License

MIT (see LICENSE).
