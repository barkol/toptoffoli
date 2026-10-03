"""Etap C (GPU Adam, potem etapB.py 'C_*.npz'): starty celowane dla k=3 (i k=2) na supportach B4:
 (1) obwod klasy 4 z klasa4_przyklad.qpy bez jednego CNOT (sasiednie warstwy scalone),
 (2) najlepsze rozwiazania kontroli k=4 (ckptB, strata<1e-9) dla danego supportu bez jednego CNOT,
 (3) bramka Margolusa (Ry(+-pi/4) na t, CX(b,t)CX(a,t)CX(b,t) i wersja a<->b), wszystkie 16 kombinacji znakow,
 (4) dla k=2: rozwiazania k=4 bez dwoch CNOT.
Kazdy start: 64 kopie z perturbacja sigma in {0,.01,.03,.1,.3,1}; Adam lr 0.01; top-8 do npz."""
import json, os, sys, itertools
import numpy as np, torch
from qiskit import qpy
from qiskit.synthesis import OneQubitEulerDecomposer
from wspolne import *
dev = 'cuda'; DEC = OneQubitEulerDecomposer('U')
def Unp(t, p, l): return np.array([[np.cos(t/2), -np.exp(1j*l)*np.sin(t/2)], [np.exp(1j*p)*np.sin(t/2), np.exp(1j*(p+l))*np.cos(t/2)]])
def drop(lay, pat, js):
    lay = [list(map(list, l)) for l in lay]; pat = list(pat)
    for j in sorted(js, reverse=True):  # CNOT j stoi miedzy warstwa j i j+1
        new = [list(DEC.angles(Unp(*lay[j + 1][q]) @ Unp(*lay[j][q]))) for q in range(3)]
        lay = lay[:j] + [new] + lay[j + 2:]; pat = pat[:j] + pat[j + 1:]
    return lay, pat
starts = []  # (name, support, pattern, layers)
c = qpy.load(open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '..', '..', 'klasa4_przyklad.qpy'), 'rb'))[0]
lay = [[None]*3]; pat = []
for ins in c.data:
    q = [c.find_bit(x).index for x in ins.qubits]
    if ins.operation.name == 'u': lay[-1][q[0]] = [float(v) for v in ins.operation.params]
    else: pat.append(tuple(q)); lay.append([None]*3)
B4 = ['B4_x0y0', 'B4_x0y1', 'B4_x1y0', 'B4_x1y1']
for s in B4:
    for j in range(4):
        l2, p2 = drop(lay, pat, [j]); starts.append((f'qpy_bez{j}', s, p2, l2))
ck = os.path.join(HERE, 'ckptB.jsonl')
sol4 = [json.loads(l) for l in open(ck)] if os.path.exists(ck) else []
for s in B4:
    good = sorted([r for r in sol4 if r['support'] == s and r['k'] == 4 and r['min_loss'] < 1e-9], key=lambda r: r['min_loss'])
    for n, r in enumerate(good[:6]):
        for j in range(4):
            l2, p2 = drop(r['argmin'], [tuple(p) for p in r['pattern']], [j]); starts.append((f'k4sol{n}_bez{j}', s, p2, l2))
        for j1, j2 in itertools.combinations(range(4), 2):
            l2, p2 = drop(r['argmin'], [tuple(p) for p in r['pattern']], [j1, j2]); starts.append((f'k4sol{n}_bez{j1}{j2}', s, p2, l2))
I = [0.0, 0.0, 0.0]
for s in B4:
    for sg in itertools.product([1, -1], repeat=4):
        for nm, pp in [('margolus', [(1, 2), (0, 2), (1, 2)]), ('margolus_ab', [(0, 2), (1, 2), (0, 2)])]:
            l = [[I, I, [sg[i] * np.pi / 4, 0.0, 0.0]] for i in range(4)]
            starts.append((f'{nm}_{"".join("+" if v > 0 else "-" for v in sg)}', s, pp, l))
print('startow', len(starts), flush=True)
SIG = torch.tensor([0.0, 0.01, 0.03, 0.1, 0.3, 1.0], dtype=torch.float64, device=dev)
NC = 64
# grupuj wedlug k, jeden batch na k
for k in sorted({len(st[2]) for st in starts}):
    grp = [st for st in starts if len(st[2]) == k]
    todo = [st for st in grp if not os.path.exists(os.path.join(HERE, f'C_{st[1]}_k{k}_{st[0]}.npz'))]
    if not todo: continue
    for s in B4:
        sub = [st for st in todo if st[1] == s]
        if not sub: continue
        L = [idx(z) for z in SUP[s]]
        base = torch.tensor([st[3] for st in sub], dtype=torch.float64, device=dev).repeat_interleave(NC, 0)
        g = torch.Generator(device=dev).manual_seed(77)
        sig = SIG[torch.arange(NC, device=dev) % len(SIG)].repeat(len(sub))
        P = (base + sig[:, None, None, None] * torch.randn(base.shape, generator=g, device=dev, dtype=torch.float64)).requires_grad_(True)
        patidx = torch.tensor([pat2idx(st[2]) for st in sub], device=dev).repeat_interleave(NC, 0)
        with torch.no_grad(): l0 = losses(P, patidx, L).reshape(len(sub), NC)
        opt = torch.optim.Adam([P], lr=0.01); ST = 2000
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ST, eta_min=1e-5)
        for it in range(ST):
            opt.zero_grad(); l = losses(P, patidx, L); l.sum().backward(); opt.step(); sch.step()
        with torch.no_grad(): lf = losses(P, patidx, L).reshape(len(sub), NC)
        Pt = P.detach().reshape(len(sub), NC, k + 1, 3, 3)
        for i, st in enumerate(sub):
            o = torch.argsort(lf[i])[:8]
            np.savez(os.path.join(HERE, f'C_{s}_k{k}_{st[0]}.npz'), pats=np.array([list(map(list, st[2]))]),
                     top=Pt[i, o].cpu().numpy()[None], loss=lf[i, o].cpu().numpy()[None], support=s, k=k, tag=st[0],
                     start_loss=float(l0[i, 0]))
        print(k, s, len(sub), 'min po Adamie', float(lf.min()), flush=True)
