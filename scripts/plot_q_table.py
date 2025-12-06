import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def find_default_qtable(project_root):
    # prefer the 10k trained table if present
    candidates = [
        os.path.join(project_root, 'q_table_trained_10k.pkl'),
        os.path.join(project_root, 'q_table_trained.pkl'),
        os.path.join(project_root, 'q_table_stage1.pkl'),
        os.path.join(project_root, 'q_table.pkl'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback: search for any .pkl in project root
    for f in os.listdir(project_root):
        if f.endswith('.pkl'):
            return os.path.join(project_root, f)
    return None


def load_qtable(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def as_array(qtable):
    # qtable may be a dict-like mapping state->list/array of action values
    if hasattr(qtable, 'items'):
        keys = list(qtable.keys())
        vals = list(qtable.values())
        # convert nested sequences to numpy arrays where possible
        try:
            arr = np.array([np.asarray(v, dtype=float) for v in vals])
        except Exception:
            # try to handle defaultdict with callable default factory
            arr = np.array([np.asarray(v) for v in vals])
        return keys, arr
    # maybe it's already a tuple
    if isinstance(qtable, (list, tuple)):
        arr = np.array(qtable)
        keys = list(range(len(arr)))
        return keys, arr
    raise ValueError('Unsupported qtable format: %s' % type(qtable))


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def plot_qtable(path, outdir, top_n=10):
    q = load_qtable(path)
    keys, arr = as_array(q)

    ensure_dir(outdir)

    # If arr is 1D (single value per state), expand
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    n_states, n_actions = arr.shape

    # Flattened distribution of all Q-values
    all_q = arr.flatten()

    plt.figure(figsize=(8, 5))
    plt.hist(all_q, bins=80, color='tab:blue', alpha=0.8)
    plt.title('Distribution of all Q-values (n={} states, {} actions)'.format(n_states, n_actions))
    plt.xlabel('Q-value')
    plt.ylabel('Count')
    out1 = os.path.join(outdir, 'q_values_hist.png')
    plt.tight_layout()
    plt.savefig(out1)
    plt.close()

    # Mean Q per action
    mean_per_action = np.nanmean(arr, axis=0)
    std_per_action = np.nanstd(arr, axis=0)

    plt.figure(figsize=(8, 5))
    x = np.arange(n_actions)
    plt.bar(x, mean_per_action, yerr=std_per_action, color='tab:green', alpha=0.9)
    plt.title('Mean Q-value per action')
    plt.xlabel('Action (column)')
    plt.ylabel('Mean Q')
    out2 = os.path.join(outdir, 'mean_q_per_action.png')
    plt.tight_layout()
    plt.savefig(out2)
    plt.close()

    # Per-state max Q histogram
    max_per_state = np.nanmax(arr, axis=1)
    plt.figure(figsize=(8, 5))
    plt.hist(max_per_state, bins=60, color='tab:orange')
    plt.title('Histogram of max Q per state')
    plt.xlabel('Max Q for state')
    plt.ylabel('Count')
    out3 = os.path.join(outdir, 'max_q_per_state_hist.png')
    plt.tight_layout()
    plt.savefig(out3)
    plt.close()

    # Top N states by max Q
    idx = np.argsort(-max_per_state)[:top_n]
    top_info = []
    for i in idx:
        key = keys[i]
        qvals = arr[i]
        top_info.append((i, key, qvals.tolist(), float(max_per_state[i])))

    # Save top states to a CSV-like text
    out_top = os.path.join(outdir, 'top_states.txt')
    with open(out_top, 'w', encoding='utf-8') as f:
        f.write('index\tstate_key\tmax_q\tq_values\n')
        for i, key, qvals, m in top_info:
            f.write(f"{i}\t{repr(key)}\t{m}\t{qvals}\n")

    print('Saved plots:')
    print(' -', out1)
    print(' -', out2)
    print(' -', out3)
    print('Top states saved to:', out_top)


def main():
    parser = argparse.ArgumentParser(description='Plot Q-table diagnostics')
    parser.add_argument('--qtable', '-q', help='Path to Q-table pickle file')
    parser.add_argument('--outdir', '-o', default='plots/q_table', help='Output directory for plots')
    parser.add_argument('--top', type=int, default=10, help='Number of top states to save')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    qpath = args.qtable or find_default_qtable(project_root)
    if qpath is None:
        print('No Q-table pickle found in project root. Provide --qtable path explicitly.')
        return

    print('Loading Q-table from:', qpath)
    plot_qtable(qpath, args.outdir, top_n=args.top)


if __name__ == '__main__':
    main()
