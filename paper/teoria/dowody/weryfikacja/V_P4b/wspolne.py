"""Wspolne definicje V_P4b. Indeks bazy x = a + 2b + 4t (a=q0,b=q1,t=q2), etykieta 'abt'.
Bramka 1q = qiskit U(th,ph,la). Obwod: warstwa0, CNOT1, warstwa1, ..., CNOTk, warstwak; P ma ksztalt (k+1,3,3)."""
import itertools, json, os
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
DT = torch.complex128
PAIRS = [(c, t) for c in range(3) for t in range(3) if c != t]
PERMT = torch.tensor([[x ^ (((x >> c) & 1) << t) for x in range(8)] for (c, t) in PAIRS])  # CNOT = inwolucja
def idx(s): return int(s[0]) + 2 * int(s[1]) + 4 * int(s[2])
def ccx(x): return x ^ 4 if (x & 1 and x & 2) else x
Q = ['100', '101', '010', '011']
SUP = {f'B4_x{x}y{y}': Q + ['00' + x, '11' + y] for x in '01' for y in '01'}
SUP['C3_ctrl'] = ['101', '010', '011', '000', '110']  # Q \ {100} u {000,110}: klasa 3
def placements(k, canon=False):
    ps = [(0, 1), (0, 2), (1, 2)] if canon else PAIRS
    return [list(s) for s in itertools.product(ps, repeat=k) if all(s[i] != s[i + 1] for i in range(k - 1))]
def u3(p):
    th, ph, la = p[..., 0], p[..., 1], p[..., 2]
    c, s = torch.cos(th / 2), torch.sin(th / 2)
    re = torch.stack([c, -torch.cos(la) * s, torch.cos(ph) * s, torch.cos(ph + la) * c], -1)
    im = torch.stack([torch.zeros_like(c), -torch.sin(la) * s, torch.sin(ph) * s, torch.sin(ph + la) * c], -1)
    return torch.complex(re, im).reshape(*p.shape[:-1], 2, 2)
def unitary(P, patidx):
    """P (B,k+1,3,3), patidx (B,k) indeksy w PAIRS -> U (B,8,8)."""
    u = u3(P); B, K = P.shape[:2]
    ut, ub, ua = u[:, :, 2], u[:, :, 1], u[:, :, 0]
    G = (ut[:, :, :, None, None, :, None, None] * ub[:, :, None, :, None, None, :, None]
         * ua[:, :, None, None, :, None, None, :]).reshape(B, K, 8, 8)
    U = G[:, 0]; perm = PERMT.to(P.device)
    for j in range(K - 1):
        pi = perm[patidx[:, j]]  # (B,8)
        U = G[:, j + 1] @ torch.gather(U, 1, pi[:, :, None].expand(-1, -1, 8))
    return U
def losses(P, patidx, L):
    U = unitary(P, patidx)
    cols = torch.tensor(L, device=P.device); rows = torch.tensor([ccx(x) for x in L], device=P.device)
    return 1 - torch.abs(U[:, rows, cols].sum(-1)) / len(L)
def pat2idx(pat): return [PAIRS.index(tuple(p)) for p in pat]
def loss_numpy(params, pat, Lstr):
    """Niezalezna kontrola: gesta macierz w numpy."""
    def U3(t, p, l): return np.array([[np.cos(t/2), -np.exp(1j*l)*np.sin(t/2)], [np.exp(1j*p)*np.sin(t/2), np.exp(1j*(p+l))*np.cos(t/2)]])
    def cn(c, t):
        M = np.zeros((8, 8)); [M.__setitem__((x ^ (((x >> c) & 1) << t), x), 1) for x in range(8)]; return M
    lay = lambda g: np.kron(np.kron(U3(*g[2]), U3(*g[1])), U3(*g[0]))
    U = lay(params[0])
    for j, (c, t) in enumerate(pat): U = lay(params[j + 1]) @ cn(c, t) @ U
    L = [idx(s) for s in Lstr]
    return 1 - abs(sum(U[ccx(x), x] for x in L)) / len(L)
