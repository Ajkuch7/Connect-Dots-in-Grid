import json
import os
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def find_results(project_root):
    files = []
    for fname in os.listdir(project_root):
        if fname.startswith('results') and fname.endswith('.json'):
            files.append(os.path.join(project_root, fname))
    return sorted(files)


def load_and_aggregate(files):
    # totals per algorithm
    totals = defaultdict(lambda: {'wins': 0, 'games': 0, 'nodes_weighted': 0.0, 'pruned_weighted': 0.0})

    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print('Failed to load', fpath, e)
            continue

        for matchup, info in data.items():
            # matchup expected like 'AlphaBeta vs Plain Minimax' or 'AlphaBeta vs Q-Learning'
            if ' vs ' in matchup:
                left, right = matchup.split(' vs ', 1)
            else:
                left = matchup
                right = 'opponent'

            report = info.get('report', {})
            episodes = int(report.get('episodes', 0) or 0)
            wins_ai = int(report.get('wins_ai', 0) or 0)
            wins_op = int(report.get('wins_opponent', 0) or 0)

            # aggregate wins/games per side
            totals[left]['wins'] += wins_ai
            totals[left]['games'] += episodes
            totals[right]['wins'] += wins_op
            totals[right]['games'] += episodes

            # weighted nodes/pruned for ai side if available
            avg_nodes = float(report.get('avg_nodes_explored_per_game') or 0)
            avg_pruned = float(report.get('avg_nodes_pruned_per_game') or 0)
            totals[left]['nodes_weighted'] += avg_nodes * episodes
            totals[left]['pruned_weighted'] += avg_pruned * episodes

    # finalize averages
    summary = {}
    for alg, v in totals.items():
        games = v['games']
        winrate = (v['wins'] / games) if games > 0 else 0.0
        avg_nodes = (v['nodes_weighted'] / games) if games > 0 else 0.0
        avg_pruned = (v['pruned_weighted'] / games) if games > 0 else 0.0
        summary[alg] = {
            'wins': v['wins'],
            'games': games,
            'winrate': winrate,
            'avg_nodes_explored_when_ai': avg_nodes,
            'avg_nodes_pruned_when_ai': avg_pruned,
        }

    return summary


def plot_summary(summary, outdir):
    ensure_dir = lambda p: os.makedirs(p, exist_ok=True) if not os.path.exists(p) else None
    ensure_dir(outdir)

    algs = sorted(summary.keys())
    winrates = [summary[a]['winrate'] for a in algs]
    games = [summary[a]['games'] for a in algs]

    x = np.arange(len(algs))
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, winrates, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:purple'])
    plt.xticks(x, algs, rotation=20)
    plt.ylim(0, 1)
    plt.ylabel('Winrate (aggregated)')
    plt.title('Aggregated Winrate per Algorithm')
    for idx, b in enumerate(bars):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{winrates[idx]:.2f}\n(n={games[idx]})", ha='center', va='bottom', fontsize=9)
    out1 = os.path.join(outdir, 'winrate_per_algorithm.png')
    plt.tight_layout()
    plt.savefig(out1)
    plt.close()

    # nodes explored when ai
    nodes = [summary[a]['avg_nodes_explored_when_ai'] for a in algs]
    plt.figure(figsize=(8, 5))
    bars2 = plt.bar(x, nodes, color='tab:gray')
    plt.xticks(x, algs, rotation=20)
    plt.ylabel('Avg nodes explored (when algorithm was AI)')
    plt.title('Average Nodes Explored per Algorithm (as AI)')
    for idx, b in enumerate(bars2):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02, f"{nodes[idx]:.0f}", ha='center', va='bottom', fontsize=9)
    out2 = os.path.join(outdir, 'nodes_explored_per_algorithm.png')
    plt.tight_layout()
    plt.savefig(out2)
    plt.close()

    # pruned (if any)
    pruned = [summary[a]['avg_nodes_pruned_when_ai'] for a in algs]
    if any(p > 0 for p in pruned):
        plt.figure(figsize=(8, 5))
        bars3 = plt.bar(x, pruned, color='tab:red')
        plt.xticks(x, algs, rotation=20)
        plt.ylabel('Avg nodes pruned (when algorithm was AI)')
        plt.title('Average Nodes Pruned per Algorithm (as AI)')
        for idx, b in enumerate(bars3):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02, f"{pruned[idx]:.1f}", ha='center', va='bottom', fontsize=9)
        out3 = os.path.join(outdir, 'nodes_pruned_per_algorithm.png')
        plt.tight_layout()
        plt.savefig(out3)
        plt.close()
    else:
        out3 = None

    print('Saved comparison plots:')
    print(' -', out1)
    print(' -', out2)
    if out3:
        print(' -', out3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-f', nargs='*', help='Optional list of result JSON files')
    parser.add_argument('--outdir', '-o', default='plots/comparison', help='Output directory')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    files = args.files or find_results(project_root)
    if not files:
        print('No results JSON files found. Run evaluations first or pass files via --files')
        return

    print('Using result files:')
    for f in files:
        print(' -', f)

    summary = load_and_aggregate(files)
    plot_summary(summary, args.outdir)


if __name__ == '__main__':
    main()
