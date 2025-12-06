# Connect-Dots-in-Grid — Connect Four

A comprehensive Connect Four implementation comparing **search-based AI** (Minimax, Alpha-Beta) against a **tabular Q-learning agent**. Includes interactive GUI, headless training/evaluation, and analysis tools.

## Overview

This project implements and compares three AI strategies for Connect Four:

1. **Alpha-Beta Pruning** — Depth-limited minimax search with pruning; efficient and strong.
2. **Plain Minimax** — Baseline minimax without pruning; slower but straightforward.
3. **Tabular Q-Learning** — Reinforcement learning agent learning from self-play; aims to generalize beyond training data.

### Key Features

- **Interactive GUI** — Play against AI or watch AI-vs-AI matches (requires `pygame`).
- **Headless training** — Train Q-learning agents via self-play against random or search-based opponents.
- **Evaluation framework** — Automatically run tournament-style matchups and export metrics (JSON, plots).
- **Analysis tools** — Visualize Q-value distributions, compare win rates, and analyze node exploration.
- **Comprehensive tests** — Unit tests for game state, win detection, and search correctness.

---

## Quick Start Guide

**New to the project?** Run these commands in order:

```powershell
# 1. Play interactively (5 seconds)
python main.py

# 2. Train Q-Learning (5 minutes)
python headless_qlearning.py --episodes 2000 --opponent random --save q_table.pkl

# 3. Compare all algorithms (1 minute)
python evaluate_algorithms.py --episodes 100 --qtable q_table.pkl --out results.json

# 4. Visualize results
python scripts/plot_comparison.py
python scripts/plot_q_table.py --qtable q_table.pkl
```

For advanced training against Minimax (longer, ~30 min):

```powershell
python headless_qlearning.py --episodes 10000 --opponent minimax --depth-op 3 --save q_table_minimax.pkl
```

---

## Installation

**Requirements**: Python 3.11+ | **Dependencies**: `pygame` ≥ 2.0, `matplotlib` ≥ 3.0

```powershell
pip install -r requirements.txt
```

---

Play against AI or watch AI-vs-AI matches:

```powershell
python main.py
```

You will be prompted to:

1. Choose an algorithm (1=AlphaBeta, 2=Minimax, 3=Q-Learning).
2. Choose a mode (1=Human vs AI, 2=AI vs AI).

**Algorithm selections**:

- `1` — **Alpha-Beta** (recommended). Fast, strong, depth=7.
- `2` — **Plain Minimax**. Slower baseline, depth=5.
- `3` — **Q-Learning**. Uses `q_table.pkl` if present; otherwise random.

**Notes**:

- Close the window to exit.
- Human plays left (AI); opponent plays right.
- Click a column to move.

#### Compare Minimax vs AlphaBeta

```powershell
python headless_runner.py
```

Outputs metrics on move time, nodes explored, and effective branching factor.

### Evaluation & Analysis

#### Run Full Tournament

Evaluate all three algorithms in matchups:

```powershell
python evaluate_algorithms.py `
  --episodes 200 `
  --qtable q_table_trained_10k.pkl `
  --out results_summary.json `
  --out-detailed results_detailed.json
```

#### Generate Q-Table Plots

```powershell
python scripts/plot_q_table.py --qtable q_table_trained_10k.pkl --outdir plots/q_table
```

Produces:

- `q_values_hist.png` — Distribution of all Q-values.
- `mean_q_per_action.png` — Mean Q for each column action.
- `max_q_per_state_hist.png` — Distribution of best Q-value per state.
- `top_states.txt` — Top states by max Q-value.

#### Compare Algorithms

```powershell
python scripts/plot_comparison.py --outdir plots/comparison
```

Produces:

- `winrate_per_algorithm.png` — Aggregated winrate per algorithm.
- `nodes_explored_per_algorithm.png` — Avg nodes when algorithm was AI.
- `nodes_pruned_per_algorithm.png` — Avg nodes pruned (pruning algorithms only).

#### Per-Matchup Analysis

```powershell
python scripts/plot_matchups.py --file results_q10k_eval.json --outdir plots/comparison
```

Produces:

- `matchup_wins.png` — Win counts (AI vs opponent) per matchup.
- `matchup_avg_nodes.png` — Avg nodes per matchup.

#### Q-Table Statistics

```powershell
python scripts/q_diagnostics.py --qtable q_table_trained_10k.pkl
```

Outputs: state count, non-zero Q-entries, Q-value statistics (mean, std, min, max).

---

### Search Metrics (Minimax & AlphaBeta)

Example output during play:

```
[AlphaBeta] Depth: 7 | Nodes: 78464 | Pruned: 154 | Time: 0.33s | Nodes/s: 234000 | EffBranch: 5.12
```

| Field         | Definition                                                                |
| ------------- | ------------------------------------------------------------------------- |
| **Depth**     | Search depth in plies (half-moves).                                       |
| **Nodes**     | Tree nodes explored.                                                      |
| **Pruned**    | Nodes skipped by alpha-beta cutoff (AlphaBeta only).                      |
| **Time**      | Wall-clock seconds.                                                       |
| **Nodes/s**   | Explored nodes per second (throughput).                                   |
| **EffBranch** | Effective branching factor = `Nodes ^ (1/Depth)`. Lower ≈ better pruning. |

**Interpretation**:

- `EffBranch` near 7 → most children explored; lower values indicate good pruning.
- Lower `Nodes` (equal strength) → more efficient search.
- `Nodes/s` measures raw speed (hardware dependent).

### Q-Learning Training Metrics

Example log:

```
Ep 500/10000  wins=53 losses=447 draws=0  eps=0.9512  time=18.4s  eval_winrate=0.065
```

| Field                 | Definition                                           |
| --------------------- | ---------------------------------------------------- |
| **Ep**                | Current episode.                                     |
| **wins/losses/draws** | Cumulative tallies during training.                  |
| **eps**               | Current epsilon (exploration rate).                  |
| **time**              | Wall-clock seconds elapsed.                          |
| **eval_winrate**      | Greedy (ε=0) winrate vs opponent in last evaluation. |

---

## Tests

Run core tests:

```powershell
python -m unittest tests.test_core -v
```

Or discover all tests:

```powershell
python -m unittest discover tests/ -v
```

**Test coverage**:

- `test_core.py` — State creation, win detection, move generation, search correctness.

---

## Quick Start Guide

### 1. Play Interactively

```powershell
python main.py
```

Choose AlphaBeta (option 1) vs Human (option 1).

### 2. Train Q-Learning (5 minutes)

```powershell
python headless_qlearning.py --episodes 2000 --opponent random --save q_table.pkl --eval-interval 500 --eval-games 100
```

### 3. Evaluate All Algorithms

```powershell
python evaluate_algorithms.py --episodes 100 --qtable q_table.pkl --out results.json
```

### 4. Visualize Results

```powershell
python scripts/plot_comparison.py
python scripts/plot_q_table.py --qtable q_table.pkl
python scripts/q_diagnostics.py --qtable q_table.pkl
```

### 5. Advanced: Train vs Minimax (takes longer)

```powershell
python headless_qlearning.py --episodes 10000 --opponent minimax --depth-op 3 --save q_table_minimax.pkl --eval-interval 500
```

---

## Troubleshooting

| Problem                   | Solution                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------ |
| GUI won't start           | Ensure `pygame` installed: `pip install pygame`. Check Python ≥ 3.11.                |
| GUI window hidden         | Check taskbar for pygame window; click to bring to front.                            |
| Q-Learning plays randomly | No Q-table saved. Run `headless_qlearning.py` to train, or GUI uses random fallback. |
| Training is slow          | Expected for minimax opponent (calls search every turn). Use `random` for warm-up.   |
| Out of memory             | Reduce `--episodes` or `--eval-games`. Q-table grows with visited states.            |
| Import errors in scripts  | Run from project root: `cd Connect-Dots-in-Grid && python scripts/...`               |
| Results JSON not created  | Ensure `evaluate_algorithms.py` runs successfully and --out flag is set.             |

---

## Project Goals & Objectives

1. **Compare search efficiency** — Demonstrate alpha-beta pruning reduces nodes by ~17× vs minimax.
2. **Evaluate Q-learning scalability** — Show limitations of tabular Q-learning on large state spaces.
3. **Teach AI concepts** — Provide clean, readable implementations of core algorithms.
4. **Enable reproducible experiments** — Support headless training, evaluation, and analysis.
5. **Visualize learning** — Plot Q-values, winrates, and node counts for understanding.

---

## Video Explanation

For a visual walkthrough of the project architecture, algorithms, and implementation details, watch this explanation:

🎥 **[Connect-Dots-in-Grid Project Walkthrough](https://youtu.be/ydysH0aNf3s)**
