#!/usr/bin/env python3
"""Visualize evaluation results from `evaluate_algorithms.py`.

Usage:
  python scripts/visualize_evaluation.py            # uses embedded sample data
  python scripts/visualize_evaluation.py --data file.json

The JSON file format should be either a mapping of matchup->stats dict, or a
list of objects with keys `matchup` and `stats`.

This script writes PNG files into `plots/` and prints their paths.
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
except Exception as e:
    print("matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
    raise


SAMPLE_DATA = {
    "AlphaBeta vs Plain Minimax": {
        "episodes": 100,
        "wins_ai": 50,
        "wins_opponent": 50,
        "draws": 0,
        "avg_moves_per_game": 7.0,
        "avg_time_per_game": 0.7540933656692504,
        "avg_nodes_explored_per_game": 259646.0,
        "avg_nodes_pruned_per_game": 1606.5,
    },
    "AlphaBeta vs Q-Learning": {
        "episodes": 100,
        "wins_ai": 99,
        "wins_opponent": 1,
        "draws": 0,
        "avg_moves_per_game": 9.39,
        "avg_time_per_game": 0.16399607181549072,
        "avg_nodes_explored_per_game": 26062.45,
        "avg_nodes_pruned_per_game": 3930.79,
    },
    "Plain Minimax vs Q-Learning": {
        "episodes": 100,
        "wins_ai": 100,
        "wins_opponent": 0,
        "draws": 0,
        "avg_moves_per_game": 9.42,
        "avg_time_per_game": 0.8824827313423157,
        "avg_nodes_explored_per_game": 410180.37,
        "avg_nodes_pruned_per_game": 0.0,
    },
}


def load_data(path: str | None) -> Dict[str, Dict[str, Any]]:
    if not path:
        return SAMPLE_DATA
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    # Normalize format
    if isinstance(loaded, dict):
        return loaded
    if isinstance(loaded, list):
        result = {}
        for obj in loaded:
            if "matchup" in obj and "stats" in obj:
                result[obj["matchup"]] = obj["stats"]
            else:
                raise ValueError("List items must be {matchup:..., stats:...}")
        return result
    raise ValueError("Unsupported JSON format")


COLOR_MAP = {
    "AlphaBeta": "tab:blue",
    "Plain Minimax": "tab:orange",
    "Q-Learning": "tab:green",
}


def parse_matchup_label(label: str) -> tuple[str, str]:
    # Expect "AlgA vs AlgB"
    parts = [p.strip() for p in label.split("vs")]
    if len(parts) == 2:
        return parts[0], parts[1]
    # fallback
    return label, ""


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_wins(data: Dict[str, Dict[str, Any]], outdir: str) -> str:
    labels = list(data.keys())
    wins_ai = [d.get("wins_ai", 0) for d in data.values()]
    wins_op = [d.get("wins_opponent", 0) for d in data.values()]
    draws = [d.get("draws", 0) for d in data.values()]

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    # draw bars per-algorithm colors
    for i, lab in enumerate(labels):
        alg_a, alg_b = parse_matchup_label(lab)
        color_a = COLOR_MAP.get(alg_a, "tab:blue")
        color_b = COLOR_MAP.get(alg_b, "tab:orange")
        ax.bar(i - width, wins_ai[i], width, color=color_a, label=(alg_a + ' wins') if i == 0 else "")
        ax.bar(i, wins_op[i], width, color=color_b, label=(alg_b + ' wins') if i == 0 else "")
        ax.bar(i + width, draws[i], width, color="gray", label=("Draws" if i == 0 else ""))
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Game counts")
    ax.set_title("Matchup results (wins/draws)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(outdir, "matchup_wins.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_summary_metrics(data: Dict[str, Dict[str, Any]], outdir: str) -> list[str]:
    labels = list(data.keys())
    avg_moves = [d.get("avg_moves_per_game", 0) for d in data.values()]
    avg_times = [d.get("avg_time_per_game", 0) for d in data.values()]

    files = []

    fig, ax = plt.subplots(figsize=(10, 5))
    # color each bar by the first algorithm in the matchup
    bar_colors = []
    for lab in labels:
        alg_a, _ = parse_matchup_label(lab)
        bar_colors.append(COLOR_MAP.get(alg_a, "tab:blue"))
    ax.bar(labels, avg_moves, color=bar_colors)
    ax.set_ylabel("Avg moves per game")
    ax.set_title("Average moves per game by matchup")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()
    f1 = os.path.join(outdir, "avg_moves_per_game.png")
    fig.savefig(f1)
    plt.close(fig)
    files.append(f1)

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = []
    for lab in labels:
        alg_a, _ = parse_matchup_label(lab)
        bar_colors.append(COLOR_MAP.get(alg_a, "tab:orange"))
    ax.bar(labels, avg_times, color=bar_colors)
    ax.set_ylabel("Avg time per game (s)")
    ax.set_title("Average time per game by matchup")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    fig.tight_layout()
    f2 = os.path.join(outdir, "avg_time_per_game.png")
    fig.savefig(f2)
    plt.close(fig)
    files.append(f2)

    return files


def plot_nodes(data: Dict[str, Dict[str, Any]], outdir: str) -> str:
    labels = list(data.keys())
    nodes = [d.get("avg_nodes_explored_per_game", 0) for d in data.values()]
    pruned = [d.get("avg_nodes_pruned_per_game", 0) for d in data.values()]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    # color nodes by algorithm A and pruned as a shade
    for i, lab in enumerate(labels):
        alg_a, _ = parse_matchup_label(lab)
        color_a = COLOR_MAP.get(alg_a, "tab:blue")
        ax.bar(i - width / 2, nodes[i], width, color=color_a)
        ax.bar(i + width / 2, pruned[i], width, color="lightgray")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Nodes (log scale)")
    ax.set_title("Avg nodes explored / pruned per game (log scale)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(outdir, "nodes_explored_pruned.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description="Visualize Connect-Dots-in-Grid evaluation results")
    p.add_argument("--data", "-d", help="Path to JSON file with evaluation results")
    p.add_argument("--outdir", "-o", default="plots", help="Directory to save plots")
    args = p.parse_args()

    data = load_data(args.data)
    ensure_dir(args.outdir)

    print("Loaded {} matchups".format(len(data)))
    out_files = []
    out_files.append(plot_wins(data, args.outdir))
    out_files.extend(plot_summary_metrics(data, args.outdir))
    out_files.append(plot_nodes(data, args.outdir))

    print("Saved plots:")
    for pth in out_files:
        print(" -", pth)


if __name__ == "__main__":
    main()
