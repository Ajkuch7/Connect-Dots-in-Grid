# Connect-Dots-in-Grid — Connect Four

A small Connect Four implementation focused on teaching and experimentation with search algorithms and simple Q-learning.

Table of contents

- [Project structure](#project-structure)
- [Requirements & install](#requirements--install)
- [Run the GUI (interactive)](#run-the-gui-interactive)
- [Headless modes (automation)](#headless-modes-automation)
- [AI metrics (explain)](#ai-metrics-explain)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Next steps / ideas](#next-steps--ideas)

## Project structure

- `core.py` — bitboard game state and search (minimax & alpha-beta)
- `main.py` — interactive GUI runner (uses `pygame`)
- `fourInARowGUI/` — GUI components and helpers
- `headless_runner.py` — automated AI-vs-AI matches for evaluation
- `q_learning.py` / `headless_qlearning.py` — tabular Q-learning agent and trainer
- `tests/` — unit tests for core logic

## Requirements & install

- Python 3.11+ (3.13 tested)
- `pygame` (only required for the GUI)

Install dependencies from the project root:

```powershell
pip install -r requirements.txt
```

## Run the GUI (interactive)

Start the GUI:

```powershell
python main.py
```

You will be prompted to select an AI algorithm and a play mode.

Algorithm choices

- `1` — Alpha-Beta (default). Depth-limited minimax with alpha-beta pruning (recommended).
- `2` — Plain Minimax. Baseline minimax without pruning.
- `3` — Q-Learning. Uses a saved `q_table.pkl` if present.

Mode choices

- `1` — Human vs AI (default).
- `2` — AI vs AI.

The GUI opens a window and animates moves; close the window to exit.

## Headless modes (automation)

Use headless scripts for experiments and training (no GUI):

```powershell
python headless_runner.py    # compare AlphaBeta vs Plain Minimax
python headless_qlearning.py # train the Q-learning agent via self-play
```

These scripts are suitable for batch runs, logging and reproducible experiments.

## AI metrics (explain)

Example metrics line printed during play:

```
[AlphaBeta] Depth: 7 | Nodes: 78464 | Pruned: 154 | Time: 0.33s | Nodes/s: 234000 | EffBranch: 5.12
```

Meaning

| Term | Description | ----------------------------------------------------------------------------------------------------------------------------------
| Depth | Search depth in half-moves (ply). Higher values explore more of the tree. |
| Nodes | Number of tree nodes expanded (visited) during the search. |
| Pruned | Count of nodes avoided by alpha-beta pruning (approximate). |
| Time | Wall-clock time spent on the search (seconds). |
| Nodes/s | Nodes explored per second (Nodes / Time). Useful to compare throughput. |
| EffBranch | Estimated effective branching factor = `Nodes ** (1.0 / Depth)` (if Depth > 0). Lower is better — indicates pruning/early cutoffs. |

Interpretation tips

- `EffBranch` near 7 => search explored most children; lower values indicate pruning or early cutoffs.
- Lower `Nodes` (for equal playing strength) => the search is more efficient.
- `Nodes/s` measures raw speed (hardware+implementation dependent).

Short example wording you can reuse:

“AlphaBeta at depth 7 explored 78,464 nodes in 0.33s (≈234k nodes/s). The effective branching factor ≈5.1, showing pruning and move ordering reduced the average branching from the theoretical maximum of 7.”

## Tests

Run the core tests I added:

```powershell
python -m unittest tests.test_core -v
```

Or run all repository tests:

```powershell
python -m unittest discover -v
```

Note: an existing test script `test_metrics.py` performs metric comparisons and may raise errors in some environments; if discover fails, run the `tests/` module directly.

## Troubleshooting

- GUI not opening: ensure `pygame` is installed in the same Python environment you use to run `main.py`.
- GUI hidden: on Windows the window may appear behind others — check the taskbar.
- For large automated experiments, prefer headless scripts instead of the GUI.
