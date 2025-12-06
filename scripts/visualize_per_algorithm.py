#!/usr/bin/env python3
"""Visualize per-episode data for each matchup from detailed JSON.

Produces per-matchup plots in `plots/per_algorithm/`:
 - time_per_episode.png
 - nodes_per_episode.png (log y)
 - cumulative_wins.png
 - hist_nodes.png
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, Any, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    print("matplotlib required. Install via pip install matplotlib", file=sys.stderr)
    raise


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def sanitize(name: str) -> str:
    # make a filesystem-safe name
    return name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_')


def plot_matchup(name: str, data: Dict[str, Any], outdir: str) -> List[str]:
    files = []
    episodes = data.get('episodes') or []
    # if top-level format is report+episodes
    if isinstance(data, dict) and 'report' in data and data['episodes'] is not None:
        episodes = data['episodes']

    if not episodes:
        return files

    nodes = [e.get('nodes_explored', 0) for e in episodes]
    times = [e.get('time', 0) for e in episodes]
    winners = [e.get('winner', 'draw') for e in episodes]

    # determine algorithm names and colors
    parts = [p.strip() for p in name.split('vs')]
    alg_a = parts[0] if parts else 'AI'
    alg_b = parts[1] if len(parts) > 1 else 'Opponent'
    COLOR_MAP = {
        'AlphaBeta': 'tab:blue',
        'Plain Minimax': 'tab:orange',
        'Q-Learning': 'tab:green',
    }
    color_a = COLOR_MAP.get(alg_a, 'tab:blue')
    color_b = COLOR_MAP.get(alg_b, 'tab:orange')

    idx = list(range(1, len(episodes) + 1))

    # time per episode
    # color points by winner: ai->alg_a color, opponent->alg_b color, draw->gray
    point_colors = [color_a if w == 'ai' else (color_b if w == 'opponent' else 'gray') for w in winners]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(idx, times, c=point_colors, edgecolors='k')
    ax.plot(idx, times, linestyle='-', color='gray', alpha=0.3)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'{name} — Time per Episode')
    fig.tight_layout()
    out = os.path.join(outdir, f'{sanitize(name)}_time_per_episode.png')
    fig.savefig(out)
    plt.close(fig)
    files.append(out)

    # nodes per episode (log)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(idx, nodes, c=point_colors, edgecolors='k')
    ax.plot(idx, nodes, linestyle='-', color='gray', alpha=0.3)
    ax.set_yscale('log')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Nodes explored')
    ax.set_title(f'{name} — Nodes per Episode (log scale)')
    fig.tight_layout()
    out = os.path.join(outdir, f'{sanitize(name)}_nodes_per_episode.png')
    fig.savefig(out)
    plt.close(fig)
    files.append(out)

    # cumulative wins
    ai_cum = []
    op_cum = []
    ai = 0
    op = 0
    for w in winners:
        if w == 'ai':
            ai += 1
        elif w == 'opponent':
            op += 1
        ai_cum.append(ai)
        op_cum.append(op)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, ai_cum, label=f'{alg_a} wins', color=color_a)
    ax.plot(idx, op_cum, label=f'{alg_b} wins', color=color_b)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative wins')
    ax.set_title(f'{name} — Cumulative wins')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(outdir, f'{sanitize(name)}_cumulative_wins.png')
    fig.savefig(out)
    plt.close(fig)
    files.append(out)

    # histogram of nodes
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nodes, bins=30, color='tab:green', log=True)
    ax.set_xlabel('Nodes explored')
    ax.set_ylabel('Count (log)')
    ax.set_title(f'{name} — Nodes histogram')
    fig.tight_layout()
    out = os.path.join(outdir, f'{sanitize(name)}_hist_nodes.png')
    fig.savefig(out)
    plt.close(fig)
    files.append(out)

    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', '-d', required=True, help='Detailed results JSON (from evaluate_algorithms.py --out-detailed)')
    p.add_argument('--outdir', '-o', default='plots/per_algorithm')
    args = p.parse_args()

    with open(args.data, 'r', encoding='utf-8') as f:
        j = json.load(f)

    ensure_dir(args.outdir)

    all_files = []
    for matchup, content in j.items():
        files = plot_matchup(matchup, content, args.outdir)
        if files:
            print(f'Wrote {len(files)} plots for {matchup}')
            all_files.extend(files)

    if all_files:
        print('Saved plots:')
        for pth in all_files:
            print(' -', pth)


if __name__ == '__main__':
    main()
