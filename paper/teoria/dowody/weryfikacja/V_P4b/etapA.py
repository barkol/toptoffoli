"""Etap A (GPU): Adam float64 dla wszystkich ustawien jednego (support,k) w jednym batchu, R startow na ustawienie.
Zapis: A_<support>_k<k>[_<tag>].npz (top-NT parametrow po Adamie na ustawienie) + linie w ckptA.jsonl. Wznawialne."""
import sys, time, json, os
import numpy as np, torch
from wspolne import *
dev = 'cuda'
def run(support, k, pats, R, steps, tag='random', P0=None, seed=0, NT=8):
    out = os.path.join(HERE, f'A_{support}_k{k}_{tag}.npz')
    if os.path.exists(out): print('skip', out); return
    t0 = time.time(); L = [idx(s) for s in SUP[support]]; nP = len(pats)
    patidx = torch.tensor([pat2idx(p) for p in pats], device=dev).repeat_interleave(R, 0)
    g = torch.Generator(device=dev).manual_seed(seed)
    if P0 is None: P0 = torch.rand(nP * R, k + 1, 3, 3, generator=g, device=dev, dtype=torch.float64) * 2 * np.pi
    P = P0.clone().requires_grad_(True)
    opt = torch.optim.Adam([P], lr=0.05); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-4)
    for it in range(steps):
        opt.zero_grad(); l = losses(P, patidx, L); l.sum().backward(); opt.step(); sch.step()
    with torch.no_grad(): lf = losses(P, patidx, L).reshape(nP, R)
    order = torch.argsort(lf, 1)[:, :NT]
    Pt = P.detach().reshape(nP, R, k + 1, 3, 3)
    top = torch.stack([Pt[i, order[i]] for i in range(nP)]).cpu().numpy()
    lt = torch.gather(lf, 1, order).cpu().numpy()
    np.savez(out, pats=np.array(pats), top=top, loss=lt, support=support, k=k, R=R, steps=steps, tag=tag)
    with open(os.path.join(HERE, 'ckptA.jsonl'), 'a') as f:
        for i, p in enumerate(pats):
            f.write(json.dumps(dict(support=support, k=k, pattern=p, tag=tag, R=R, steps=steps,
                                    adam_min=float(lt[i, 0]), adam_top=[float(v) for v in lt[i]])) + '\n')
    print(support, k, tag, 'min', float(lt.min()), 'czas', round(time.time() - t0), 's', flush=True)
if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 256; ST = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    B4 = ['B4_x0y0', 'B4_x0y1', 'B4_x1y0', 'B4_x1y1']
    for s in B4: run(s, 4, placements(4, canon=True), R, ST, seed=11)      # kontrola dodatnia
    run('C3_ctrl', 3, placements(3), R, ST, seed=12)                       # kontrola dodatnia
    for n, s in enumerate(B4): run(s, 3, placements(3), R, ST, seed=20 + n)
    for n, s in enumerate(B4): run(s, 2, placements(2), R, ST, seed=30 + n)
