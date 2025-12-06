#!/usr/bin/env python3
"""Create a scatter plot of avg nodes explored vs avg time per game.

Usage:
  python scripts/visualize_point.py --data results.json --outdir plots
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    print("matplotlib required. Install via pip install matplotlib", file=sys.stderr)
    raise


def load(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", "-d", required=True, help="Path to results.json")
    p.add_argument("--outdir", "-o", default="plots")
    args = p.parse_args()

    data = load(args.data)
    ensure_dir(args.outdir)

    labels = list(data.keys())
    nodes = [data[k].get("avg_nodes_explored_per_game", 0) for k in labels]
    times = [data[k].get("avg_time_per_game", 0) for k in labels]

    # color mapping per algorithm (choose color of the first algorithm in the matchup)
    COLOR_MAP = {
        "AlphaBeta": "tab:blue",
        "Plain Minimax": "tab:orange",
        "Q-Learning": "tab:green",
    }

    def parse_matchup(lab: str):
        parts = [p.strip() for p in lab.split('vs')]
        return parts[0] if parts else lab

    colors = [COLOR_MAP.get(parse_matchup(l), 'tab:gray') for l in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(nodes, times, s=120, c=colors, edgecolors="k")
    for i, lab in enumerate(labels):
        ax.annotate(lab, (nodes[i], times[i]), xytext=(5, 5), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Avg nodes explored per game (log scale)")
    ax.set_ylabel("Avg time per game (s)")
    ax.set_title("Nodes vs Time per matchup")
    fig.tight_layout()

    out = os.path.join(args.outdir, "point_scatter.png")
    fig.savefig(out)
    print("Saved", out)


if __name__ == '__main__':
    main()
