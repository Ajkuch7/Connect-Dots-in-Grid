import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_matchups(data):
    matchups = []
    for m, info in data.items():
        report = info.get('report', {})
        episodes = int(report.get('episodes', 0) or 0)
        wins_ai = int(report.get('wins_ai', 0) or 0)
        wins_op = int(report.get('wins_opponent', 0) or 0)
        avg_nodes = float(report.get('avg_nodes_explored_per_game') or 0)
        avg_pruned = float(report.get('avg_nodes_pruned_per_game') or 0)
        matchups.append({'matchup': m, 'wins_ai': wins_ai, 'wins_op': wins_op, 'episodes': episodes,
                         'avg_nodes': avg_nodes, 'avg_pruned': avg_pruned})
    return matchups


def plot_winrates(matchups, outpath):
    labels = [m['matchup'] for m in matchups]
    ai_wins = [m['wins_ai'] for m in matchups]
    op_wins = [m['wins_op'] for m in matchups]
    episodes = [m['episodes'] for m in matchups]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, ai_wins, width, label='AI wins')
    plt.bar(x + width/2, op_wins, width, label='Opponent wins')
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel('Wins')
    plt.title('Wins per Matchup')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_nodes(matchups, outpath):
    labels = [m['matchup'] for m in matchups]
    avg_nodes = [m['avg_nodes'] for m in matchups]
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(x, avg_nodes, color='tab:gray')
    plt.xticks(x, labels, rotation=30, ha='right')
    plt.ylabel('Avg nodes explored (AI)')
    plt.title('Average Nodes Explored per Matchup (AI side)')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default='results_q10k_eval.json')
    parser.add_argument('--outdir', '-o', default='plots/comparison')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print('Results file not found:', args.file)
        return
    data = load_results(args.file)
    matchups = collect_matchups(data)
    os.makedirs(args.outdir, exist_ok=True)
    out1 = os.path.join(args.outdir, 'matchup_wins.png')
    out2 = os.path.join(args.outdir, 'matchup_avg_nodes.png')
    plot_winrates(matchups, out1)
    plot_nodes(matchups, out2)
    print('Saved:', out1)
    print('Saved:', out2)


if __name__ == '__main__':
    main()
