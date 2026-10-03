"""Etap B (CPU): L-BFGS-B float64 (scipy, gradient z torch) z top-NT punktow etapu A, dla kazdego (support,k,placement).
Checkpoint ckptB.jsonl, linia na (support,k,placement,tag); wznawialne. Uzycie: etapB.py NPROC [glob]"""
import sys, glob, json, os, time
import numpy as np, torch
from multiprocessing import Pool
from scipy.optimize import minimize
from wspolne import *
torch.set_num_threads(1)
def refine(args):
    support, k, pat, tag, tops = args
    L = [idx(s) for s in SUP[support]]; pi = torch.tensor([pat2idx(pat)])
    def f(x):
        P = torch.tensor(x.reshape(1, k + 1, 3, 3), requires_grad=True)
        l = losses(P, pi, L)[0]; l.backward(); return float(l), P.grad.numpy().ravel().copy()
    best = (np.inf, None); res_all = []
    for x0 in tops:
        r = minimize(f, x0.ravel(), jac=True, method='L-BFGS-B', options=dict(maxiter=5000, ftol=1e-16, gtol=1e-13, maxcor=50))
        res_all.append(float(r.fun))
        if r.fun < best[0]: best = (float(r.fun), r.x.reshape(k + 1, 3, 3))
    chk = loss_numpy(best[1], pat, SUP[support])
    return dict(key=f'{support}|{k}|{json.dumps(pat)}|{tag}', support=support, k=k, pattern=pat, tag=tag,
                min_loss=best[0], loss_numpy=float(chk), refined=sorted(res_all), argmin=best[1].tolist())
if __name__ == '__main__':
    NP = int(sys.argv[1]); pat_glob = sys.argv[2] if len(sys.argv) > 2 else 'A_*.npz'
    ck = os.path.join(HERE, 'ckptB.jsonl')
    done = {json.loads(l)['key'] for l in open(ck)} if os.path.exists(ck) else set()
    tasks = []
    for fn in sorted(glob.glob(os.path.join(HERE, pat_glob))):
        d = np.load(fn); s, k, tag = str(d['support']), int(d['k']), str(d['tag'])
        for i, p in enumerate(d['pats'].tolist()):
            if f'{s}|{k}|{json.dumps(p)}|{tag}' not in done: tasks.append((s, k, p, tag, d['top'][i]))
    print('zadan', len(tasks), flush=True); t0 = time.time()
    with Pool(NP) as pool:
        for n, rec in enumerate(pool.imap_unordered(refine, tasks)):
            with open(ck, 'a') as f: f.write(json.dumps(rec) + '\n')
            if n % 25 == 0: print(n, rec['key'], rec['min_loss'], round(time.time() - t0), flush=True)
