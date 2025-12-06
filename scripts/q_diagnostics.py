import pickle, os
import numpy as np

def main(path='q_table_trained_10k.pkl'):
    if not os.path.exists(path):
        print('Q-table not found:', path)
        return
    with open(path,'rb') as f:
        q = pickle.load(f)
    n_states = len(q)
    vals = [v for v in q.values()]
    arr = np.array([np.array(v, dtype=float) for v in vals])
    allq = arr.flatten()
    nonzero = (allq != 0).sum()
    print('Q-table path:', path)
    print('states =', n_states)
    print('total Q entries =', allq.size)
    print('nonzero Q entries =', nonzero, f'({nonzero/allq.size*100:.2f}%)')
    print('mean Q =', float(allq.mean()), 'std =', float(allq.std()))
    print('max Q =', float(allq.max()), 'min Q =', float(allq.min()))
    print('mean of max-per-state =', float(np.max(arr, axis=1).mean()))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--qtable', '-q', default='q_table_trained_10k.pkl')
    args = p.parse_args()
    main(args.qtable)
